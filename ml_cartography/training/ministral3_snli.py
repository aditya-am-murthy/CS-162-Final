"""SNLI cartography for Ministral 3 / Unsloth 4-bit checkpoints (text-only path)."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Dict, List, Optional

_nullcontext = contextlib.nullcontext

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

# text-only tokenizer ids (avoid Unsloth/Pixtral processor treating SNLI as images)
_MINISTRAL_TEXT_TOKENIZER_IDS = {
    "3b": "mistralai/Ministral-3-3B-Base-2512",
    "8b": "mistralai/Ministral-3-8B-Base-2512",
    "14b": "mistralai/Ministral-3-14B-Base-2512",
}


def is_ministral3_model(model_name: str) -> bool:
    n = model_name.lower()
    return "ministral" in n or ("unsloth" in n and "mistral" in n)


def _resolve_text_tokenizer_id(model_name: str) -> str:
    n = model_name.lower()
    if "14b" in n:
        return _MINISTRAL_TEXT_TOKENIZER_IDS["14b"]
    if "8b" in n:
        return _MINISTRAL_TEXT_TOKENIZER_IDS["8b"]
    return _MINISTRAL_TEXT_TOKENIZER_IDS["3b"]


def _load_text_only_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok_id = _resolve_text_tokenizer_id(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
    ensure_padding_token(tokenizer)
    return tokenizer


def _as_text_tokenizer(tokenizer):
    """Unsloth may return a multimodal Processor; use underlying text tokenizer."""
    cls = type(tokenizer).__name__.lower()
    if "processor" in cls or "pixtral" in cls:
        inner = getattr(tokenizer, "tokenizer", None)
        if inner is not None:
            return inner
    if hasattr(tokenizer, "tokenizer") and type(tokenizer).__name__ != "PreTrainedTokenizerFast":
        inner = getattr(tokenizer, "tokenizer", None)
        if inner is not None and callable(inner):
            return inner
    return tokenizer


def _tokenize_snli(tokenizer, text: str, max_length: int) -> Dict:
    tok = _as_text_tokenizer(tokenizer)
    return tok(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )


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
        enc = _tokenize_snli(self.tokenizer, text, self.max_length)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long),
            "guid": row["guid"],
        }


def _backbone_param_device(backbone: nn.Module) -> torch.device:
    try:
        return next(backbone.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _backbone_compute_dtype(backbone: nn.Module) -> torch.dtype:
    for p in backbone.parameters():
        if p.dtype.is_floating_point:
            return p.dtype
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _prepare_backbone_for_snli(backbone: nn.Module, freeze: bool) -> None:
    if hasattr(backbone, "gradient_checkpointing_disable"):
        backbone.gradient_checkpointing_disable()
    if hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()


def _align_classifier_to_backbone(wrapper: "SnliClassifierWrapper") -> None:
    """4-bit Unsloth: head must match backbone device + compute dtype (bf16/fp16)."""
    dev = _backbone_param_device(wrapper.backbone)
    dtype = _backbone_compute_dtype(wrapper.backbone)
    wrapper.classifier.to(device=dev, dtype=dtype)


class SnliClassifierWrapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int = 3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.classifier = nn.Linear(hidden_size, num_labels)
        _align_classifier_to_backbone(self)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _forward_backbone(self, input_ids, attention_mask):
        bb = self.backbone
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        try:
            return bb(**kwargs)
        except TypeError:
            pass
        lm = getattr(bb, "language_model", None)
        if lm is not None:
            return lm(**kwargs)
        inner = getattr(bb, "model", None)
        if inner is not None and inner is not bb:
            return inner(**kwargs)
        return bb(**kwargs)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        ctx = torch.no_grad() if self.freeze_backbone else _nullcontext()
        with ctx:
            out = self._forward_backbone(input_ids, attention_mask)
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            h = out.last_hidden_state
        elif getattr(out, "hidden_states", None):
            h = out.hidden_states[-1]
        else:
            raise RuntimeError("backbone did not return hidden states; check model class")

        if self.freeze_backbone:
            h = h.detach()

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(dtype=h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = h[:, -1, :]

        pooled = pooled.to(
            device=self.classifier.weight.device,
            dtype=self.classifier.weight.dtype,
        )
        logits = self.classifier(pooled)
        if labels is not None:
            labels = labels.to(logits.device)
        loss = F.cross_entropy(logits.float(), labels) if labels is not None else None
        return _ModelOutput(loss=loss, logits=logits.float())


class _ModelOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


def _load_backbone_and_tokenizer(model_name: str):
    """Load 4-bit weights via Unsloth; always tokenize with text-only Mistral tokenizer."""
    tokenizer = _load_text_only_tokenizer(model_name)

    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            "ministral-3b requires unsloth to load the Unsloth 4-bit checkpoint. "
            "Install with: pip install unsloth"
        ) from e

    model, _ = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    ensure_padding_token(tokenizer, model)
    hidden = resolve_config_hidden_size(model.config)
    return model, tokenizer, hidden, True, "unsloth"


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

    backbone, tokenizer, hidden_size, quantized, _loader = _load_backbone_and_tokenizer(
        cfg.model_name
    )
    freeze_bb = cfg.ministral_freeze_backbone
    _prepare_backbone_for_snli(backbone, freeze=freeze_bb)
    model = SnliClassifierWrapper(
        backbone, hidden_size, num_labels=3, freeze_backbone=freeze_bb
    )
    if not quantized:
        model.to(device)
    else:
        _align_classifier_to_backbone(model)
    mode = "frozen backbone + trainable head" if freeze_bb else "full finetune"
    print(
        f"ministral3_snli ({mode}): backbone {_backbone_param_device(backbone)} "
        f"{_backbone_compute_dtype(backbone)}, classifier {model.classifier.weight.device} "
        f"{model.classifier.weight.dtype}"
    )

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
        train_params = [p for p in model.classifier.parameters() if p.requires_grad]
        if not freeze_bb:
            train_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(train_params, lr=cfg.learning_rate)
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
