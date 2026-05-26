"""Fine-tune Hugging Face models on GLUE-style tasks and log training dynamics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ml_cartography.training.dynamic_cartography import (
    append_training_metric,
    curriculum_weights_from_coordinates,
    guid_weights_to_sample_weights,
    records_to_coordinates,
    save_snapshot,
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
    # dynamic maps (Idea #2)
    dynamic_snapshots: bool = True
    snapshot_dir: Optional[Path] = None
    curriculum_after_epoch: int = 0
    curriculum_ambiguous_boost: float = 2.5
    curriculum_easy_scale: float = 0.4
    # colab / large causal models
    load_in_4bit: bool = False
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1


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


MODEL_PRESETS = {
    "distilbert": "distilbert-base-uncased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "bert-base": "bert-base-uncased",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "ministral-3b": "unsloth/Ministral-3-3B-Base-2512-unsloth-bnb-4bit",
    # legacy alias
    "mistral-7b": "unsloth/Ministral-3-3B-Base-2512-unsloth-bnb-4bit",
}

# colab T4 friendly defaults per preset
PRESET_DEFAULTS: Dict[str, Dict] = {
    "distilbert": {"batch_size": 32, "max_length": 128, "load_in_4bit": False},
    "roberta-base": {"batch_size": 16, "max_length": 128, "load_in_4bit": False},
    "llama-3.2-1b": {
        "batch_size": 4,
        "max_length": 256,
        "load_in_4bit": True,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
    },
    "ministral-3b": {
        "batch_size": 4,
        "max_length": 256,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
    },
    "mistral-7b": {
        "batch_size": 4,
        "max_length": 256,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
    },
}


def apply_preset_defaults(cfg: TrainConfig, preset: Optional[str]) -> TrainConfig:
    if not preset or preset not in PRESET_DEFAULTS:
        return cfg
    defaults = PRESET_DEFAULTS[preset]
    for key, val in defaults.items():
        if not hasattr(cfg, key):
            continue
        current = getattr(cfg, key)
        # only override generic constructor defaults for large-model keys
        if key in ("batch_size", "max_length", "load_in_4bit", "gradient_checkpointing", "gradient_accumulation_steps", "learning_rate"):
            setattr(cfg, key, val)
    return cfg


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


def _is_prequantized_checkpoint(model_name: str) -> bool:
    n = model_name.lower()
    return "bnb-4bit" in n or "unsloth" in n and "4bit" in n


def _load_model_and_tokenizer(cfg: TrainConfig, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict = {"num_labels": num_labels}
    if cfg.load_in_4bit and not _is_prequantized_checkpoint(cfg.model_name):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        **model_kwargs,
    )
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer


def _build_train_loader(
    train_ds: NliDataset,
    cfg: TrainConfig,
    sample_weights: Optional[List[float]],
    device: torch.device,
) -> DataLoader:
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
    return DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    grad_accum: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc=f"train epoch {epoch}", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss / grad_accum
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += float(loss.item()) * grad_accum
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
    metrics_log: Optional[Path] = None,
) -> Dict:
    """
    Fine-tune on SNLI (train split), log gold-label probability after each epoch.
    Optional dynamic snapshots + curriculum reweighting (Idea #2).
    """
    if cfg.dataset.lower() != "snli":
        raise ValueError("only dataset=snli is implemented in glue_trainer")

    from ml_cartography.training.ministral3_snli import (
        is_ministral3_model,
        train_and_collect_dynamics_ministral3,
    )

    if is_ministral3_model(cfg.model_name):
        return train_and_collect_dynamics_ministral3(cfg, wandb_run, metrics_log)

    torch.manual_seed(cfg.seed)
    device = _resolve_device()
    use_fp16 = cfg.fp16 and device.type == "cuda" and not cfg.load_in_4bit
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    grad_accum = max(1, cfg.gradient_accumulation_steps)

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
    model, tokenizer = _load_model_and_tokenizer(cfg, num_labels)
    if not cfg.load_in_4bit:
        model.to(device)

    train_ds = NliDataset(train_rows, tokenizer, cfg.max_length)
    val_ds = NliDataset(val_rows, tokenizer, cfg.max_length)
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

    train_guids = [r["guid"] for r in train_rows]
    sample_weights: Optional[List[float]] = None
    snapshot_dir = cfg.snapshot_dir
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    cfg.output_logs.parent.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict] = []
    snapshot_paths: List[Tuple[int, Path]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loader = _build_train_loader(train_ds, cfg, sample_weights, device)
        total_steps = (len(train_loader) // grad_accum) * (cfg.epochs - epoch + 1)
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, max(total_steps, 1)
        )

        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler,
            epoch,
            grad_accum=grad_accum,
        )
        val_acc = _evaluate_accuracy(model, val_loader, device)
        epoch_records = _collect_epoch_predictions(
            model, collect_loader, device, epoch
        )
        all_records.extend(epoch_records)

        if cfg.dynamic_snapshots and snapshot_dir:
            coords = records_to_coordinates(all_records, max_epoch=epoch)
            snap_path = save_snapshot(coords, snapshot_dir, epoch)
            snapshot_paths.append((epoch, snap_path))

        metric_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "num_prediction_rows": len(epoch_records),
        }
        if metrics_log:
            append_training_metric(metrics_log, metric_row)

        if wandb_run is not None:
            import wandb

            log_payload = dict(metric_row)
            if snapshot_dir and (snapshot_dir / f"epoch_{epoch:03d}_coordinates.jsonl").is_file():
                from ml_cartography.analysis.data_map import save_data_map_plot

                plot_path = snapshot_dir / f"epoch_{epoch:03d}_data_map.png"
                coords = records_to_coordinates(all_records, max_epoch=epoch)
                save_data_map_plot(coords, plot_path)
                log_payload[f"data_map_epoch_{epoch}"] = wandb.Image(str(plot_path))
            wandb.log(log_payload, step=epoch)

        if cfg.checkpoint_dir:
            ckpt = cfg.checkpoint_dir / f"epoch-{epoch}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)

        # adaptive curriculum for remaining epochs (Idea #2)
        if (
            cfg.curriculum_after_epoch > 0
            and epoch >= cfg.curriculum_after_epoch
            and epoch < cfg.epochs
        ):
            coords = records_to_coordinates(all_records, max_epoch=epoch)
            guid_w = curriculum_weights_from_coordinates(
                coords,
                ambiguous_boost=cfg.curriculum_ambiguous_boost,
                easy_scale=cfg.curriculum_easy_scale,
            )
            sample_weights = guid_weights_to_sample_weights(train_guids, guid_w)

    with cfg.output_logs.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row) + "\n")

    if cfg.checkpoint_dir:
        final_dir = cfg.checkpoint_dir.parent / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

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
        "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
        "num_snapshots": len(snapshot_paths),
    }
    return summary
