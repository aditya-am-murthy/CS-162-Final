"""Log training dynamics on preference pairs (Idea #1: Preference Data Maps)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml_cartography.training.dynamic_cartography import (
    append_training_metric,
    save_snapshot,
)
from ml_cartography.core.preference_dynamics import (
    preference_epoch_rows_to_coordinates,
)
from ml_cartography.analysis.preference_map import (
    annotate_preference_regions,
    save_preference_map_plot,
)


@dataclass
class PreferenceTrainConfig:
    model_name: str = "distilbert-base-uncased"
    max_samples: Optional[int] = 5000
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 256
    seed: int = 42
    fp16: bool = True
    output_logs: Path = Path("data/raw/preference_epoch_predictions.jsonl")
    snapshot_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None


class PreferencePairDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        chosen_enc = self.tokenizer(
            row["prompt"],
            row["chosen"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            row["prompt"],
            row["rejected"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "guid": row["guid"],
        }


def _load_preference_rows(max_samples: Optional[int], seed: int) -> List[Dict]:
    """Load preference pairs from UltraFeedback (chosen/rejected) or synthetic fallback."""
    rows: List[Dict] = []
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceH4/ultrafeedback_binarized",
            split="train_prefs",
            streaming=True,
        )
        for i, ex in enumerate(ds):
            if max_samples is not None and len(rows) >= max_samples:
                break
            prompt = ex.get("prompt") or ex.get("instruction") or ""
            chosen_msgs = ex.get("chosen", [])
            rejected_msgs = ex.get("rejected", [])
            if isinstance(chosen_msgs, list) and chosen_msgs:
                chosen = chosen_msgs[-1].get("content", "") if isinstance(chosen_msgs[-1], dict) else str(chosen_msgs[-1])
            else:
                chosen = str(chosen_msgs)
            if isinstance(rejected_msgs, list) and rejected_msgs:
                rejected = rejected_msgs[-1].get("content", "") if isinstance(rejected_msgs[-1], dict) else str(rejected_msgs[-1])
            else:
                rejected = str(rejected_msgs)
            if not prompt or not chosen or not rejected:
                continue
            rows.append(
                {
                    "guid": f"pref-ultra-{i:07d}",
                    "prompt": prompt[:2000],
                    "chosen": chosen[:1500],
                    "rejected": rejected[:1500],
                }
            )
    except Exception as e:
        print(f"ultrafeedback load failed ({e}), using synthetic preference pairs")
        import random

        rng = random.Random(seed)
        templates = [
            ("Explain photosynthesis.", "Plants convert light to chemical energy via chlorophyll.", "Photosynthesis is when animals eat plants only."),
            ("What is 2+2?", "4", "22"),
            ("Write a polite refusal for an unsafe request.", "I can't help with that, but I can suggest safe alternatives.", "Sure, here is how to do the harmful thing."),
        ]
        for i in range(max_samples or 500):
            t = templates[i % len(templates)]
            noise = rng.random() * 0.02
            rows.append(
                {
                    "guid": f"pref-synth-{i:07d}",
                    "prompt": t[0],
                    "chosen": t[1] + (" " if noise > 0.01 else ""),
                    "rejected": t[2],
                }
            )
    return rows


def _pair_scores(model, batch, device) -> tuple:
    c_logits = model(
        input_ids=batch["chosen_input_ids"].to(device),
        attention_mask=batch["chosen_attention_mask"].to(device),
    ).logits.squeeze(-1)
    r_logits = model(
        input_ids=batch["rejected_input_ids"].to(device),
        attention_mask=batch["rejected_attention_mask"].to(device),
    ).logits.squeeze(-1)
    margin = c_logits - r_logits
    prob_chosen = torch.sigmoid(margin)
    return prob_chosen, margin


@torch.no_grad()
def _collect_preference_epoch(model, loader, device, epoch) -> List[Dict]:
    model.eval()
    out_rows: List[Dict] = []
    for batch in tqdm(loader, desc=f"pref collect {epoch}", leave=False):
        prob, margin = _pair_scores(model, batch, device)
        for i, guid in enumerate(batch["guid"]):
            out_rows.append(
                {
                    "guid": guid,
                    "epoch": epoch,
                    "prob_chosen": round(float(prob[i].item()), 6),
                    "reward_margin": round(float(margin[i].item()), 6),
                }
            )
    return out_rows


def train_and_collect_preference_dynamics(
    cfg: PreferenceTrainConfig,
    wandb_run=None,
    metrics_log: Optional[Path] = None,
) -> Dict:
    from ml_cartography.training.glue_trainer import _resolve_device

    device = _resolve_device()
    rows = _load_preference_rows(cfg.max_samples, cfg.seed)
    if not rows:
        raise ValueError("no preference rows loaded")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=1
    )
    model.to(device)

    ds = PreferencePairDataset(rows, tokenizer, cfg.max_length)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    collect_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    cfg.output_logs.parent.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"pref train {epoch}", leave=False):
            prob, margin = _pair_scores(model, batch, device)
            targets = torch.ones_like(prob)
            loss = torch.nn.functional.binary_cross_entropy(prob, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n += 1

        epoch_rows = _collect_preference_epoch(model, collect_loader, device, epoch)
        all_records.extend(epoch_rows)

        if cfg.snapshot_dir:
            coords = preference_epoch_rows_to_coordinates(all_records)
            for row in coords:
                from ml_cartography.analysis.preference_map import assign_preference_region
                row["region"] = assign_preference_region(
                    float(row["confidence"]), float(row["variability"])
                )
            save_snapshot(coords, cfg.snapshot_dir, epoch)
            plot_path = cfg.snapshot_dir / f"epoch_{epoch:03d}_preference_map.png"
            tagged = annotate_preference_regions(coords)
            save_preference_map_plot(tagged, plot_path)
            if wandb_run is not None:
                import wandb
                wandb.log(
                    {
                        "epoch": epoch,
                        "pref_train_loss": total_loss / max(n, 1),
                        "preference_map": wandb.Image(str(plot_path)),
                    },
                    step=epoch,
                )
        elif wandb_run is not None:
            import wandb
            wandb.log({"epoch": epoch, "pref_train_loss": total_loss / max(n, 1)}, step=epoch)

        if metrics_log:
            append_training_metric(
                metrics_log,
                {"epoch": epoch, "pref_train_loss": total_loss / max(n, 1)},
            )

    with cfg.output_logs.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row) + "\n")

    if cfg.checkpoint_dir:
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(cfg.checkpoint_dir)
        tokenizer.save_pretrained(cfg.checkpoint_dir)

    return {
        "model_name": cfg.model_name,
        "num_pairs": len(rows),
        "epochs": cfg.epochs,
        "output_logs": str(cfg.output_logs),
        "num_log_rows": len(all_records),
    }
