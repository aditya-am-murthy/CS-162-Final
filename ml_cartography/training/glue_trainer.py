"""Fine-tune Hugging Face models on GLUE-style tasks and log training dynamics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
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
    # ministral + unsloth 4-bit: full backbone FT often dtype/checkpoint errors on T4
    ministral_freeze_backbone: bool = True
    winogrande_config: str = "winogrande_xl"


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


def _load_mnli_rows(
    split: str,
    max_samples: Optional[int],
    subset_guids: Optional[Set[str]],
    seed: int,
) -> List[Dict]:
    """MultiNLI (Williams et al., 2018) — 3-way NLI, matched validation split."""
    from datasets import load_dataset

    hf_split = "train" if split == "train" else "validation_matched"
    ds = load_dataset("nyu-mll/glue", "mnli", split=hf_split)
    ds = ds.filter(lambda x: x["label"] != -1)

    rows: List[Dict] = []
    for i, ex in enumerate(ds):
        guid = f"mnli-{hf_split}-{i:07d}"
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


def _load_qnli_rows(
    split: str,
    max_samples: Optional[int],
    subset_guids: Optional[Set[str]],
    seed: int,
) -> List[Dict]:
    """QNLI — binary entailment (question, passage sentence), from SQuAD-derived GLUE task."""
    from datasets import load_dataset

    hf_split = "train" if split == "train" else "validation"
    ds = load_dataset("nyu-mll/glue", "qnli", split=hf_split)

    rows: List[Dict] = []
    for i, ex in enumerate(ds):
        guid = f"qnli-{hf_split}-{i:07d}"
        if subset_guids is not None and guid not in subset_guids:
            continue
        rows.append(
            {
                "guid": guid,
                "premise": ex["question"],
                "hypothesis": ex["sentence"],
                "label": int(ex["label"]),
            }
        )

    if max_samples is not None and len(rows) > max_samples:
        rng = __import__("random").Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]
    return rows


def _fill_winogrande_blank(sentence: str, option: str) -> str:
    """Fill the first `_` slot (SuperGLUE / paper-style)."""
    idx = sentence.index("_")
    return sentence[:idx] + option + sentence[idx + 1 :]


def _winogrande_gold_index(answer: str) -> int:
    return 0 if str(answer).strip() == "1" else 1


def _load_winogrande_rows(
    split: str,
    max_samples: Optional[int],
    subset_guids: Optional[Set[str]],
    seed: int,
    config_name: str = "winogrande_xl",
) -> List[Dict]:
    """
    Raw WinoGrande items (one row per prompt).

  Training uses per-option binary rows (see _expand_winogrande_training_rows).
    """
    from datasets import load_dataset

    hf_split = "train" if split == "train" else "validation"
    ds = load_dataset("allenai/winogrande", config_name, split=hf_split)

    rows: List[Dict] = []
    for i, ex in enumerate(ds):
        guid = f"winogrande-{hf_split}-{i:07d}"
        if subset_guids is not None and guid not in subset_guids:
            continue
        rows.append(
            {
                "guid": guid,
                "sentence": ex["sentence"],
                "option1": ex["option1"],
                "option2": ex["option2"],
                "answer": str(ex["answer"]).strip(),
            }
        )

    if max_samples is not None and len(rows) > max_samples:
        rng = __import__("random").Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]
    return rows


class WinograndePairDataset(Dataset):
    """One row per WinoGrande item; loss compares both filled sentences."""

    def __init__(self, raw_rows: List[Dict]):
        self.rows = raw_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        sent = row["sentence"]
        return {
            "guid": row["guid"],
            "text1": _fill_winogrande_blank(sent, row["option1"]),
            "text2": _fill_winogrande_blank(sent, row["option2"]),
            "label": _winogrande_gold_index(row["answer"]),
        }


def _collate_winogrande_pairs(batch: List[Dict], tokenizer, max_length: int) -> Dict:
    texts1 = [b["text1"] for b in batch]
    texts2 = [b["text2"] for b in batch]
    enc1 = tokenizer(
        texts1,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    enc2 = tokenizer(
        texts2,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return {
        "input_ids1": enc1["input_ids"],
        "attention_mask1": enc1["attention_mask"],
        "input_ids2": enc2["input_ids"],
        "attention_mask2": enc2["attention_mask"],
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "guids": [b["guid"] for b in batch],
    }


def _winogrande_pair_logits(
    model,
    input_ids1: torch.Tensor,
    attention_mask1: torch.Tensor,
    input_ids2: torch.Tensor,
    attention_mask2: torch.Tensor,
) -> torch.Tensor:
    """Per-item scores for option1 vs option2 (log-odds of class 1 on each fill)."""
    logits1 = model(input_ids=input_ids1, attention_mask=attention_mask1).logits
    logits2 = model(input_ids=input_ids2, attention_mask=attention_mask2).logits
    s1 = logits1[:, 1] - logits1[:, 0]
    s2 = logits2[:, 1] - logits2[:, 0]
    return torch.stack([s1, s2], dim=1)


def _load_dataset_rows(
    dataset: str,
    split: str,
    max_samples: Optional[int],
    subset_guids: Optional[Set[str]],
    seed: int,
    winogrande_config: str = "winogrande_xl",
) -> List[Dict]:
    d = dataset.lower()
    if d == "snli":
        return _load_snli_rows(split, max_samples, subset_guids, seed)
    if d == "mnli":
        return _load_mnli_rows(split, max_samples, subset_guids, seed)
    if d == "qnli":
        return _load_qnli_rows(split, max_samples, subset_guids, seed)
    if d == "winogrande":
        return _load_winogrande_rows(
            split, max_samples, subset_guids, seed, config_name=winogrande_config
        )
    raise ValueError(f"unsupported dataset: {dataset}")


def dataset_num_labels(dataset: str) -> int:
    d = dataset.lower()
    if d in ("snli", "mnli"):
        return 3
    if d in ("qnli", "winogrande"):
        return 2
    raise ValueError(f"unsupported dataset: {dataset}")


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


def ensure_padding_token(tokenizer, model=None) -> None:
    """Llama/Mistral need pad_token_id on tokenizer + model.config for batch > 1."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(tokenizer, "pad_token_id", None) is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    if model is not None and getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "model") and hasattr(model.model, "config"):
            model.model.config.pad_token_id = tokenizer.pad_token_id


def resolve_config_hidden_size(config) -> int:
    if hasattr(config, "hidden_size") and config.hidden_size is not None:
        return int(config.hidden_size)
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return int(text_cfg.hidden_size)
    if hasattr(config, "hidden_sizes") and config.hidden_sizes:
        return int(config.hidden_sizes[0])
    raise AttributeError(f"cannot find hidden_size on {type(config).__name__}")


def _load_model_and_tokenizer(cfg: TrainConfig, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    ensure_padding_token(tokenizer)

    model_kwargs: Dict = {"num_labels": num_labels}
    if tokenizer.pad_token_id is not None:
        model_kwargs["pad_token_id"] = tokenizer.pad_token_id
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
    ensure_padding_token(tokenizer, model)
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


def _build_winogrande_pair_loader(
    train_ds: WinograndePairDataset,
    cfg: TrainConfig,
    tokenizer,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=lambda b: _collate_winogrande_pairs(b, tokenizer, cfg.max_length),
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

    optimizer_steps = max(0, n_batches // grad_accum)
    if n_batches > 0 and n_batches % grad_accum != 0:
        optimizer_steps += 1
    return total_loss / max(n_batches, 1), optimizer_steps


def _train_epoch_winogrande(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    cfg: TrainConfig,
    grad_accum: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc=f"train epoch {epoch}", leave=False)):
        input_ids1 = batch["input_ids1"].to(device)
        attention_mask1 = batch["attention_mask1"].to(device)
        input_ids2 = batch["input_ids2"].to(device)
        attention_mask2 = batch["attention_mask2"].to(device)
        labels = batch["labels"].to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pair_logits = _winogrande_pair_logits(
                    model, input_ids1, attention_mask1, input_ids2, attention_mask2
                )
                loss = F.cross_entropy(pair_logits, labels) / grad_accum
            scaler.scale(loss).backward()
        else:
            pair_logits = _winogrande_pair_logits(
                model, input_ids1, attention_mask1, input_ids2, attention_mask2
            )
            loss = F.cross_entropy(pair_logits, labels) / grad_accum
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

    optimizer_steps = max(0, n_batches // grad_accum)
    if n_batches > 0 and n_batches % grad_accum != 0:
        optimizer_steps += 1
    return total_loss / max(n_batches, 1), optimizer_steps


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


PAPER_STYLE_DATASETS = frozenset({"snli", "mnli", "qnli", "winogrande"})


@torch.no_grad()
def _evaluate_accuracy(model, loader: DataLoader, device: torch.device) -> float:
    if loader is None:
        raise ValueError("val_loader is None; use dataset-specific evaluation (e.g. WinoGrande)")
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


@torch.no_grad()
def _collect_winogrande_epoch_predictions(
    model,
    tokenizer,
    raw_rows: List[Dict],
    device: torch.device,
    epoch: int,
    max_length: int,
) -> List[Dict]:
    """One dynamics row per WinoGrande item using pairwise option probabilities."""
    model.eval()
    records: List[Dict] = []
    loader = DataLoader(
        WinograndePairDataset(raw_rows),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=lambda b: _collate_winogrande_pairs(b, tokenizer, max_length),
    )
    for batch in tqdm(loader, desc=f"collect epoch {epoch}", leave=False):
        pair_logits = _winogrande_pair_logits(
            model,
            batch["input_ids1"].to(device),
            batch["attention_mask1"].to(device),
            batch["input_ids2"].to(device),
            batch["attention_mask2"].to(device),
        )
        pair_probs = torch.softmax(pair_logits, dim=-1)
        preds = pair_probs.argmax(dim=-1)
        labels = batch["labels"]
        guids = batch["guids"]
        for i in range(len(guids)):
            gold = int(labels[i].item())
            pred = int(preds[i].item())
            prob_gold = float(pair_probs[i, gold].item())
            records.append(
                {
                    "guid": guids[i],
                    "epoch": epoch,
                    "gold_label": gold,
                    "pred_label": pred,
                    "prob_gold": round(prob_gold, 6),
                }
            )
    return records


@torch.no_grad()
def _evaluate_winogrande_accuracy(
    model,
    tokenizer,
    raw_rows: List[Dict],
    device: torch.device,
    max_length: int,
) -> float:
    if not raw_rows:
        return 0.0
    correct = 0
    total = 0
    loader = DataLoader(
        WinograndePairDataset(raw_rows),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=lambda b: _collate_winogrande_pairs(b, tokenizer, max_length),
    )
    for batch in loader:
        pair_logits = _winogrande_pair_logits(
            model,
            batch["input_ids1"].to(device),
            batch["attention_mask1"].to(device),
            batch["input_ids2"].to(device),
            batch["attention_mask2"].to(device),
        )
        preds = pair_logits.argmax(dim=-1).cpu()
        labels = batch["labels"]
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))
    return correct / max(total, 1)


def train_and_collect_dynamics(
    cfg: TrainConfig,
    wandb_run=None,
    metrics_log: Optional[Path] = None,
) -> Dict:
    """
    Fine-tune on SNLI or WinoGrande, log gold-label probability after each epoch.
    Optional dynamic snapshots + curriculum reweighting (Idea #2).
    """
    if cfg.dataset.lower() not in ("snli", "mnli", "qnli", "winogrande"):
        raise ValueError("dataset must be snli, mnli, qnli, or winogrande")

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

    is_winogrande = cfg.dataset.lower() == "winogrande"
    if is_winogrande and cfg.max_length == 128:
        cfg.max_length = 256

    train_raw = _load_dataset_rows(
        cfg.dataset,
        "train",
        cfg.max_train_samples,
        cfg.subset_guids,
        cfg.seed,
        winogrande_config=cfg.winogrande_config,
    )
    val_raw = _load_dataset_rows(
        cfg.dataset,
        "validation",
        cfg.max_eval_samples,
        None,
        cfg.seed,
        winogrande_config=cfg.winogrande_config,
    )
    if not train_raw:
        raise ValueError("no training examples after filtering; check subset guids")

    num_labels = 2 if is_winogrande else dataset_num_labels(cfg.dataset)
    model, tokenizer = _load_model_and_tokenizer(cfg, num_labels)
    if not cfg.load_in_4bit:
        model.to(device)

    if is_winogrande:
        train_ds = WinograndePairDataset(train_raw)
        val_loader = None
        collect_loader = None
        train_guids = [r["guid"] for r in train_raw]
    else:
        train_ds = NliDataset(train_raw, tokenizer, cfg.max_length)
        val_ds = NliDataset(val_raw, tokenizer, cfg.max_length)
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
        train_guids = [r["guid"] for r in train_raw]
    sample_weights: Optional[List[float]] = None
    snapshot_dir = cfg.snapshot_dir
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    cfg.output_logs.parent.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict] = []
    snapshot_paths: List[Tuple[int, Path]] = []
    prev_snapshot_coords: Optional[List[Dict]] = None
    cumulative_optimizer_steps = 0
    cumulative_param_units = 0.0
    idea2_metric_history: List[Dict] = []
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )
    if is_winogrande:
        total_batches_per_epoch = max((len(train_ds) + cfg.batch_size - 1) // cfg.batch_size, 1)
    else:
        total_batches_per_epoch = max(len(train_ds) // cfg.batch_size, 1)
        if len(train_ds) % cfg.batch_size != 0:
            total_batches_per_epoch += 1
    total_optimizer_steps = max(
        ((total_batches_per_epoch + grad_accum - 1) // grad_accum) * cfg.epochs,
        1,
    )
    warmup_steps = int(total_optimizer_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_optimizer_steps,
    )

    for epoch in range(1, cfg.epochs + 1):
        if is_winogrande:
            train_loader = _build_winogrande_pair_loader(train_ds, cfg, tokenizer, device)
        else:
            train_loader = _build_train_loader(train_ds, cfg, sample_weights, device)

        if is_winogrande:
            train_loss, optimizer_steps = _train_epoch_winogrande(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                scaler,
                epoch,
                cfg,
                grad_accum=grad_accum,
            )
        else:
            train_loss, optimizer_steps = _train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                scaler,
                epoch,
                grad_accum=grad_accum,
            )
        cumulative_optimizer_steps += optimizer_steps
        param_units = optimizer_steps * trainable_params
        cumulative_param_units += param_units
        if is_winogrande:
            val_acc = _evaluate_winogrande_accuracy(
                model, tokenizer, val_raw, device, cfg.max_length
            )
            epoch_records = _collect_winogrande_epoch_predictions(
                model, tokenizer, train_raw, device, epoch, cfg.max_length
            )
        else:
            val_acc = _evaluate_accuracy(model, val_loader, device)
            epoch_records = _collect_epoch_predictions(
                model, collect_loader, device, epoch
            )
        all_records.extend(epoch_records)

        coords = records_to_coordinates(all_records, max_epoch=epoch)
        if cfg.dynamic_snapshots and snapshot_dir:
            snap_path = save_snapshot(coords, snapshot_dir, epoch)
            snapshot_paths.append((epoch, snap_path))

        movement_log: Dict[str, float] = {}
        if cfg.dynamic_snapshots:
            from ml_cartography.analysis.movement_metrics import (
                compute_epoch_movement,
                compute_learnability_efficiency,
                save_learnability_vs_compute_plot,
                save_transition_heatmap,
                strip_internal_keys,
            )

            movement = compute_epoch_movement(prev_snapshot_coords, coords)
            transition = movement.pop("_transition_matrix", None)
            delta_learn = float(
                movement.get("learnability/delta", movement.get("learnability/index", 0.0))
            )
            movement.update(
                compute_learnability_efficiency(
                    delta_learnability=delta_learn,
                    optimizer_steps=optimizer_steps,
                    trainable_params=trainable_params,
                    batch_size=cfg.batch_size,
                    seq_length=cfg.max_length,
                )
            )
            movement["compute/cumulative_optimizer_steps"] = float(cumulative_optimizer_steps)
            movement["compute/cumulative_param_update_units"] = float(cumulative_param_units)
            idea2_metric_history.append(dict(movement))
            movement_log = strip_internal_keys(movement)
            if transition is not None and epoch > 1 and snapshot_dir:
                save_transition_heatmap(
                    transition,
                    snapshot_dir / f"epoch_{epoch:03d}_region_transition.png",
                    title=f"Region transitions → epoch {epoch}",
                )
            if snapshot_dir:
                save_learnability_vs_compute_plot(
                    idea2_metric_history,
                    snapshot_dir / "learnability_vs_compute.png",
                )
            prev_snapshot_coords = coords

        metric_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "num_prediction_rows": len(epoch_records),
            "compute/optimizer_steps_epoch": optimizer_steps,
            "compute/cumulative_optimizer_steps": cumulative_optimizer_steps,
            "compute/cumulative_param_update_units": cumulative_param_units,
            **movement_log,
        }
        if metrics_log:
            append_training_metric(metrics_log, metric_row)

        if wandb_run is not None:
            import wandb

            log_payload = dict(metric_row)
            if snapshot_dir:
                from ml_cartography.analysis.data_map import save_data_map_plot

                plot_path = snapshot_dir / f"epoch_{epoch:03d}_data_map.png"
                paper_style = cfg.dataset.lower() in PAPER_STYLE_DATASETS
                map_title = f"{cfg.dataset.upper()} data map (through epoch {epoch})"
                save_data_map_plot(
                    coords,
                    plot_path,
                    color_by="correctness" if paper_style else "region",
                    title=map_title,
                )
                log_payload[f"data_map_epoch_{epoch}"] = wandb.Image(str(plot_path))
                if epoch > 1:
                    heatmap = snapshot_dir / f"epoch_{epoch:03d}_region_transition.png"
                    if heatmap.is_file():
                        log_payload[f"idea2/region_transition_epoch_{epoch}"] = wandb.Image(
                            str(heatmap)
                        )
                eff_plot = snapshot_dir / "learnability_vs_compute.png"
                if eff_plot.is_file():
                    log_payload["idea2/learnability_vs_compute"] = wandb.Image(str(eff_plot))
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

    if device.type == "cpu":
        print(
            "WARNING: training on CPU — very slow. On Colab: T4 runtime + "
            "scripts/colab_setup.sh (do not pip install -r requirements-train.txt)."
        )
    else:
        try:
            print(f"using device: {device} ({torch.cuda.get_device_name(0)})")
        except Exception:
            print(f"using device: {device}")
    summary = {
        "device": str(device),
        "dataset": cfg.dataset,
        "model_name": cfg.model_name,
        "num_train": len(train_raw),
        "num_val": len(val_raw),
        "epochs": cfg.epochs,
        "final_val_accuracy": (
            _evaluate_winogrande_accuracy(model, tokenizer, val_raw, device, cfg.max_length)
            if is_winogrande
            else _evaluate_accuracy(model, val_loader, device)
        ),
        "output_logs": str(cfg.output_logs),
        "num_log_rows": len(all_records),
        "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
        "num_snapshots": len(snapshot_paths),
    }
    return summary
