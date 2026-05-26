"""SNLI cartography for Ministral 3 / Unsloth 4-bit checkpoints (text-only path)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ml_cartography.training.dynamic_cartography import (
    append_training_metric,
    curriculum_weights_from_coordinates,
    guid_weights_to_sample_weights,
    records_to_coordinates,
    save_snapshot,
)
from ml_cartography.training.glue_trainer import (
    TrainConfig,
    _build_train_loader,
    _load_snli_rows,
    _resolve_device,
    _train_epoch,
    ensure_padding_token,
    resolve_config_hidden_size,
)


def is_ministral3_model(model_name: str) -> bool:
    n = model_name.lower()
    return "ministral" in n or ("unsloth" in n and "mistral" in n)


class _SnliTextDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        text = (
            f"Premise: {row['premise']}\n"
            f"Hypothesis: {row['hypothesis']}\n"
            f"Label:"
        )
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long),
            "guid": row["guid"],
        }


class SnliClassifierWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int = 3):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            h = out.last_hidden_state
        elif getattr(out, "hidden_states", None):
            h = out.hidden_states[-1]
        else:
            raise RuntimeError("backbone did not return hidden states; check model class")

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = h[:, -1, :]

        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return _ModelOutput(loss=loss, logits=logits)


class _ModelOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


def _load_backbone_and_tokenizer(model_name: str):
    """Prefer Unsloth loader; fall back to transformers AutoModel."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        ensure_padding_token(tokenizer, model)
        hidden = resolve_config_hidden_size(model.config)
        return model, tokenizer, hidden, True
    except ImportError:
        pass

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ensure_padding_token(tokenizer)

    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    ensure_padding_token(tokenizer, model)
    hidden = resolve_config_hidden_size(model.config)
    return model, tokenizer, hidden, True


@torch.no_grad()
def _collect_epoch_predictions(model, loader, device, epoch) -> List[Dict]:
    model.eval()
    records: List[Dict] = []
    for batch in tqdm(loader, desc=f"collect epoch {epoch}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        guids = batch["guid"]
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(out.logits, dim=-1)
        preds = probs.argmax(dim=-1)
        for i in range(len(guids)):
            gold = int(labels[i].item())
            records.append(
                {
                    "guid": guids[i],
                    "epoch": epoch,
                    "gold_label": gold,
                    "pred_label": int(preds[i].item()),
                    "prob_gold": round(float(probs[i, gold].item()), 6),
                }
            )
    return records


@torch.no_grad()
def _evaluate_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = out.logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))
    return correct / max(total, 1)


def train_and_collect_dynamics_ministral3(
    cfg: TrainConfig,
    wandb_run=None,
    metrics_log: Optional[Path] = None,
) -> Dict:
    torch.manual_seed(cfg.seed)
    device = _resolve_device()
    grad_accum = max(1, cfg.gradient_accumulation_steps)

    train_rows = _load_snli_rows(
        "train", cfg.max_train_samples, cfg.subset_guids, cfg.seed
    )
    val_rows = _load_snli_rows("validation", cfg.max_eval_samples, None, cfg.seed)
    if not train_rows:
        raise ValueError("no training examples after filtering")

    backbone, tokenizer, hidden_size, quantized = _load_backbone_and_tokenizer(
        cfg.model_name
    )
    model = SnliClassifierWrapper(backbone, hidden_size, num_labels=3)
    if not quantized:
        model.to(device)

    train_ds = _SnliTextDataset(train_rows, tokenizer, cfg.max_length)
    val_ds = _SnliTextDataset(val_rows, tokenizer, cfg.max_length)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False)
    collect_loader = DataLoader(train_ds, batch_size=cfg.eval_batch_size, shuffle=False)

    train_guids = [r["guid"] for r in train_rows]
    sample_weights = None
    snapshot_dir = cfg.snapshot_dir
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    cfg.output_logs.parent.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict] = []
    use_fp16 = cfg.fp16 and device.type == "cuda" and not quantized
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    for epoch in range(1, cfg.epochs + 1):
        train_loader = _build_train_loader(train_ds, cfg, sample_weights, device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
        )
        from torch.optim.lr_scheduler import LambdaLR

        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        train_loss = _train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, epoch, grad_accum
        )
        val_acc = _evaluate_accuracy(model, val_loader, device)
        epoch_records = _collect_epoch_predictions(model, collect_loader, device, epoch)
        all_records.extend(epoch_records)

        if cfg.dynamic_snapshots and snapshot_dir:
            coords = records_to_coordinates(all_records, max_epoch=epoch)
            save_snapshot(coords, snapshot_dir, epoch)

        if metrics_log:
            append_training_metric(
                metrics_log,
                {"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc},
            )
        if wandb_run is not None:
            import wandb

            wandb.log(
                {"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc},
                step=epoch,
            )

        if (
            cfg.curriculum_after_epoch > 0
            and epoch >= cfg.curriculum_after_epoch
            and epoch < cfg.epochs
        ):
            coords = records_to_coordinates(all_records, max_epoch=epoch)
            guid_w = curriculum_weights_from_coordinates(coords)
            sample_weights = guid_weights_to_sample_weights(train_guids, guid_w)

    with cfg.output_logs.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row) + "\n")

    return {
        "device": str(device),
        "model_name": cfg.model_name,
        "backend": "ministral3_snli",
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "epochs": cfg.epochs,
        "final_val_accuracy": _evaluate_accuracy(model, val_loader, device),
        "output_logs": str(cfg.output_logs),
        "num_log_rows": len(all_records),
    }
