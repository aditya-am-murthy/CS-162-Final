"""Instruction-tuning dynamics on Alpaca-style data (extends cartography to IT)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ml_cartography.training.dynamic_cartography import append_training_metric, save_snapshot
from ml_cartography.training.glue_trainer import records_to_coordinates, _resolve_device
from ml_cartography.analysis.data_map import save_data_map_plot


@dataclass
class InstructionTrainConfig:
    model_name: str = "distilbert-base-uncased"
    max_samples: Optional[int] = 3000
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 256
    seed: int = 42
    output_logs: Path = Path("data/raw/instruction_epoch_predictions.jsonl")
    snapshot_dir: Optional[Path] = None


class InstructionDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        text = (
            f"### Instruction:\n{row['instruction']}\n"
            f"### Input:\n{row.get('input', '')}\n"
            f"### Response:\n{row['output']}"
        )
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        prompt_len = len(
            self.tokenizer(
                f"### Instruction:\n{row['instruction']}\n### Input:\n{row.get('input', '')}\n### Response:\n",
                add_special_tokens=False,
            )["input_ids"]
        )
        labels[0, :prompt_len] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "guid": row["guid"],
        }


def _load_instruction_rows(max_samples: Optional[int], seed: int) -> List[Dict]:
    rows: List[Dict] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        for i, ex in enumerate(ds):
            if max_samples is not None and len(rows) >= max_samples:
                break
            rows.append(
                {
                    "guid": f"alpaca-{i:07d}",
                    "instruction": ex["instruction"],
                    "input": ex.get("input") or "",
                    "output": ex["output"],
                }
            )
    except Exception as e:
        print(f"alpaca load failed ({e}), using synthetic instructions")
        for i in range(max_samples or 200):
            rows.append(
                {
                    "guid": f"alpaca-synth-{i:07d}",
                    "instruction": "Summarize the following.",
                    "input": "Dataset cartography maps examples by training dynamics.",
                    "output": "Cartography uses confidence and variability across epochs.",
                }
            )
    return rows


def _response_token_prob(model, batch, device) -> torch.Tensor:
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )
    # proxy for confidence: negative mean loss on response tokens
    loss = out.loss
    prob_proxy = torch.exp(-loss.detach().clamp(max=10))
    return prob_proxy


@torch.no_grad()
def _collect_instruction_epoch(model, loader, device, epoch) -> List[Dict]:
    model.eval()
    records: List[Dict] = []
    for batch in tqdm(loader, desc=f"it collect {epoch}", leave=False):
        prob = _response_token_prob(model, batch, device)
        for i, guid in enumerate(batch["guid"]):
            p = float(prob.item()) if prob.dim() == 0 else float(prob[i].item())
            records.append(
                {
                    "guid": guid,
                    "epoch": epoch,
                    "gold_label": 0,
                    "pred_label": 0,
                    "prob_gold": round(p, 6),
                }
            )
    return records


def train_and_collect_instruction_dynamics(
    cfg: InstructionTrainConfig,
    wandb_run=None,
    metrics_log: Optional[Path] = None,
) -> Dict:
    device = _resolve_device()
    rows = _load_instruction_rows(cfg.max_samples, cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.to(device)

    ds = InstructionDataset(rows, tokenizer, cfg.max_length)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    collect_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    all_records: List[Dict] = []
    cfg.output_logs.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"it train {epoch}", leave=False):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n += 1

        epoch_rows = _collect_instruction_epoch(model, collect_loader, device, epoch)
        all_records.extend(epoch_rows)

        if cfg.snapshot_dir:
            coords = records_to_coordinates(all_records, max_epoch=epoch)
            save_snapshot(coords, cfg.snapshot_dir, epoch)
            plot_path = cfg.snapshot_dir / f"epoch_{epoch:03d}_instruction_map.png"
            save_data_map_plot(coords, plot_path)
            if wandb_run is not None:
                import wandb
                wandb.log(
                    {
                        "epoch": epoch,
                        "it_loss": total_loss / max(n, 1),
                        "instruction_map": wandb.Image(str(plot_path)),
                    },
                    step=epoch,
                )

        if metrics_log:
            append_training_metric(metrics_log, {"epoch": epoch, "it_loss": total_loss / max(n, 1)})

    with cfg.output_logs.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row) + "\n")

    return {
        "model_name": cfg.model_name,
        "num_examples": len(rows),
        "epochs": cfg.epochs,
        "output_logs": str(cfg.output_logs),
    }
