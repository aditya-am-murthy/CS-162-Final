"""LoRA/QLoRA causal LM training for SNLI + instruction-style tasks (Llama, Mistral)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from ml_cartography.training.dynamic_cartography import (
    append_training_metric,
    curriculum_weights_from_coordinates,
    guid_weights_to_sample_weights,
    records_to_coordinates,
    save_snapshot,
)
from ml_cartography.training.experiment_run import ExperimentPaths
from ml_cartography.training.glue_trainer import _load_snli_rows, _resolve_device


LABEL_NAMES = ["entailment", "neutral", "contradiction"]

CAUSAL_PRESETS = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}


@dataclass
class CausalTrainConfig:
    model_name: str
    task: str = "snli"  # snli | alpaca
    max_train_samples: Optional[int] = 8000
    max_eval_samples: Optional[int] = 1000
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_length: int = 256
    seed: int = 42
    use_lora: bool = True
    load_in_4bit: bool = True
    map_interval: int = 1
    curriculum_enabled: bool = False
    paths: Optional[ExperimentPaths] = None


def _format_snli_prompt(premise: str, hypothesis: str, label_name: Optional[str] = None) -> str:
    base = (
        "Classify the relationship between premise and hypothesis as "
        "entailment, neutral, or contradiction.\n\n"
        f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer:"
    )
    if label_name:
        return base + " " + label_name
    return base


def _load_alpaca_rows(max_samples: Optional[int], seed: int) -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rows = []
    for i, ex in enumerate(ds):
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "") or ""
        output = ex.get("output", "")
        prompt = instruction if not inp else f"{instruction}\n{inp}"
        rows.append(
            {
                "guid": f"alpaca-{i:07d}",
                "prompt": prompt.strip(),
                "response": output.strip(),
                "label": 0,
            }
        )
    if max_samples and len(rows) > max_samples:
        import random

        random.Random(seed).shuffle(rows)
        rows = rows[:max_samples]
    return rows


class CausalDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int, task: str):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        if self.task == "snli":
            label_name = LABEL_NAMES[int(row["label"])]
            text = _format_snli_prompt(row["premise"], row["hypothesis"], label_name)
            gold_label = int(row["label"])
        else:
            text = f"### Instruction:\n{row['prompt']}\n\n### Response:\n{row['response']}"
            gold_label = 0
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "guid": row["guid"],
            "gold_label": gold_label,
            "premise": row.get("premise", ""),
            "hypothesis": row.get("hypothesis", ""),
            "prompt": row.get("prompt", ""),
        }


def _build_causal_model(cfg: CausalTrainConfig, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if cfg.load_in_4bit and device.type == "cuda":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    if "device_map" not in model_kwargs:
        model.to(device)
    return model, tokenizer


@torch.no_grad()
def _score_snli_labels(model, tokenizer, row: Dict, device, max_length: int) -> tuple:
    import math

    logps = []
    for name in LABEL_NAMES:
        text = _format_snli_prompt(row["premise"], row["hypothesis"], name)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        out = model(**enc, labels=enc["input_ids"])
        logps.append(-float(out.loss.item()))
    max_lp = max(logps)
    probs = [math.exp(lp - max_lp) for lp in logps]
    z = sum(probs)
    probs = [p / z for p in probs]
    pred = int(max(range(3), key=lambda i: probs[i]))
    gold = int(row["label"])
    return gold, pred, float(probs[gold])


@torch.no_grad()
def _collect_causal_epoch(
    model,
    tokenizer,
    rows: List[Dict],
    device: torch.device,
    epoch: int,
    task: str,
    max_length: int,
) -> List[Dict]:
    model.eval()
    records = []
    for row in tqdm(rows, desc=f"causal collect {epoch}", leave=False):
        if task == "snli":
            gold, pred, prob_gold = _score_snli_labels(model, tokenizer, row, device, max_length)
        else:
            lp = _mean_response_logprob(model, tokenizer, row["prompt"], row["response"], device, max_length)
            prob_gold = 1.0 / (1.0 + __import__("math").exp(-lp))
            gold, pred = 0, 1 if prob_gold > 0.5 else 0
        records.append(
            {
                "guid": row["guid"],
                "epoch": epoch,
                "gold_label": gold,
                "pred_label": pred,
                "prob_gold": round(prob_gold, 6),
            }
        )
    return records


def _mean_response_logprob(model, tokenizer, prompt, response, device, max_length) -> float:
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
        return -float(out.loss.item())


def train_causal_dynamics(cfg: CausalTrainConfig, wandb_run=None) -> Dict:
    device = _resolve_device()
    paths = cfg.paths

    if cfg.task == "snli":
        train_rows = _load_snli_rows("train", cfg.max_train_samples, None, cfg.seed)
        val_rows = _load_snli_rows("validation", cfg.max_eval_samples, None, cfg.seed)
    else:
        train_rows = _load_alpaca_rows(cfg.max_train_samples, cfg.seed)
        val_rows = train_rows[: min(500, len(train_rows))]

    model, tokenizer = _build_causal_model(cfg, device)
    train_ds = CausalDataset(train_rows, tokenizer, cfg.max_length, cfg.task)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )
    total_steps = max(len(train_loader) * cfg.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.06 * total_steps), total_steps
    )

    all_records: List[Dict] = []
    snapshots: List = []
    guid_order = [r["guid"] for r in train_rows]

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"causal train {epoch}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(out.loss.item())
            n_batches += 1

        epoch_records = _collect_causal_epoch(
            model, tokenizer, train_rows, device, epoch, cfg.task, cfg.max_length
        )
        all_records.extend(epoch_records)

        if paths:
            append_training_metric(
                paths.training_metrics_path(),
                {"epoch": epoch, "train_loss": total_loss / max(n_batches, 1)},
            )
            if epoch % cfg.map_interval == 0:
                coords = records_to_coordinates(all_records, max_epoch=epoch)
                snap = save_snapshot(coords, paths.snapshots_dir, epoch)
                snapshots.append((epoch, snap))

        if wandb_run:
            import wandb

            wandb.log({"epoch": epoch, "loss": total_loss / max(n_batches, 1)}, step=epoch)

        if paths:
            ckpt = paths.checkpoints_dir / f"epoch-{epoch}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)

    out_logs = paths.epoch_logs_path() if paths else Path("data/raw/epoch_predictions_causal.jsonl")
    with out_logs.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    coords = records_to_coordinates(all_records)
    if paths:
        from ml_cartography.analysis.data_map import annotate_regions
        from ml_cartography.utils.io import write_jsonl

        write_jsonl(paths.coordinates_path(), coords)
        write_jsonl(paths.regions_path(), annotate_regions(coords))
        if snapshots:
            from ml_cartography.training.dynamic_cartography import build_region_trajectories

            write_jsonl(paths.trajectories_path(), build_region_trajectories(snapshots))
        model.save_pretrained(paths.models_dir)
        tokenizer.save_pretrained(paths.models_dir)

    return {
        "task": cfg.task,
        "model_name": cfg.model_name,
        "num_train": len(train_rows),
        "epochs": cfg.epochs,
        "output_logs": str(out_logs),
        "device": str(device),
    }
