"""Fine-tune Hugging Face models on GLUE-style tasks and log training dynamics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


@dataclass
class TrainConfig:
    dataset: str = "snli"
    model_name: str = "distilbert-base-uncased"
    max_train_samples: Optional[int] = 20000
    max_eval_samples: Optional[int] = 2000
    epochs: int = 5
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    max_length: int = 128
    warmup_ratio: float = 0.06
    seed: int = 42
    fp16: bool = True
    output_logs: Path = Path("data/raw/epoch_predictions_trained.jsonl")
    checkpoint_dir: Optional[Path] = None
    subset_guids: Optional[Set[str]] = None


class NliDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        enc = self.tokenizer(
            row["premise"],
            row["hypothesis"],
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


def _load_snli_rows(
    split: str,
    max_samples: Optional[int],
    subset_guids: Optional[Set[str]],
    seed: int,
) -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/snli", split=split)
    ds = ds.filter(lambda x: x["label"] != -1)

    rows: List[Dict] = []
    for i, ex in enumerate(ds):
        guid = f"snli-{split}-{i:07d}"
        if subset_guids is not None and guid not in subset_guids:
            continue
        rows.append(
            {
                "guid": guid,
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": int(ex["label"]),
            }
        )

    if max_samples is not None and len(rows) > max_samples:
        rng = __import__("random").Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]
    return rows


def load_guids_from_jsonl(path: Path) -> Set[str]:
    guids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                guids.add(json.loads(line)["guid"])
    return guids


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            pass
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc=f"train epoch {epoch}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss
            loss.backward()
            optimizer.step()

        scheduler.step()
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _collect_epoch_predictions(
    model,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> List[Dict]:
    model.eval()
    records: List[Dict] = []

    for batch in tqdm(loader, desc=f"collect epoch {epoch}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        guids = batch["guid"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        for i in range(len(guids)):
            gold = int(labels[i].item())
            pred = int(preds[i].item())
            records.append(
                {
                    "guid": guids[i],
                    "epoch": epoch,
                    "gold_label": gold,
                    "pred_label": pred,
                    "prob_gold": round(float(probs[i, gold].item()), 6),
                }
            )
    return records


@torch.no_grad()
def _evaluate_accuracy(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))
    return correct / max(total, 1)


def train_and_collect_dynamics(
    cfg: TrainConfig,
    wandb_run=None,
) -> Dict:
    """
    Fine-tune on SNLI (train split), log gold-label probability after each epoch.

    Returns summary dict with paths and final validation accuracy.
    """
    if cfg.dataset.lower() != "snli":
        raise ValueError("only dataset=snli is implemented; more can be added later")

    torch.manual_seed(cfg.seed)
    device = _resolve_device()
    use_fp16 = cfg.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    train_rows = _load_snli_rows(
        "train",
        cfg.max_train_samples,
        cfg.subset_guids,
        cfg.seed,
    )
    val_rows = _load_snli_rows(
        "validation",
        cfg.max_eval_samples,
        None,
        cfg.seed,
    )
    if not train_rows:
        raise ValueError("no training examples after filtering; check subset guids")

    num_labels = 3
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_labels
    )
    model.to(device)

    train_ds = NliDataset(train_rows, tokenizer, cfg.max_length)
    val_ds = NliDataset(val_rows, tokenizer, cfg.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    collect_loader = DataLoader(
        train_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    cfg.output_logs.parent.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, epoch
        )
        val_acc = _evaluate_accuracy(model, val_loader, device)
        epoch_records = _collect_epoch_predictions(
            model, collect_loader, device, epoch
        )
        all_records.extend(epoch_records)

        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_accuracy": val_acc,
                    "num_prediction_rows": len(epoch_records),
                },
                step=epoch,
            )

        if cfg.checkpoint_dir:
            ckpt = cfg.checkpoint_dir / f"epoch-{epoch}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)

    with cfg.output_logs.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row) + "\n")

    print(f"using device: {device}")
    summary = {
        "device": str(device),
        "model_name": cfg.model_name,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "epochs": cfg.epochs,
        "final_val_accuracy": _evaluate_accuracy(model, val_loader, device),
        "output_logs": str(cfg.output_logs),
        "num_log_rows": len(all_records),
    }
    return summary


MODEL_PRESETS = {
    "distilbert": "distilbert-base-uncased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "bert-base": "bert-base-uncased",
}
