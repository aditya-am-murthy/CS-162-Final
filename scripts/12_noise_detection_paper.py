#!/usr/bin/env python3
"""
Paper §5: label-noise injection and automatic mislabeled-example detection.

This script implements the WinoGrande experiment described in the paper:
  1. choose 1% of train examples from the easy-to-learn region,
  2. flip their labels and retrain a model on the noised train set,
  3. recompute cartography coordinates after retraining,
  4. train a confidence-only linear detector on balanced noisy/clean examples,
  5. run that detector on the original cartography coordinates.

Without --train, the script prepares the flip files and a retraining command.
With --train, it runs the retraining and detector end to end.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml_cartography.utils.io import read_jsonl, write_json, write_jsonl
from scripts.common import (
    add_wandb_args,
    finish_wandb,
    init_wandb,
    load_hf_credentials,
    load_pipeline_config,
    use_wandb,
)


def _num_labels(dataset: str) -> int:
    if dataset in ("qnli", "winogrande"):
        return 2
    if dataset in ("snli", "mnli"):
        return 3
    raise ValueError(f"unsupported dataset: {dataset}")


def _flip_label(label: int, dataset: str, rng: random.Random) -> int:
    n_labels = _num_labels(dataset)
    if n_labels == 2:
        return 1 - int(label)
    choices = [i for i in range(n_labels) if i != int(label)]
    return rng.choice(choices)


def _easy_candidates(rows: list[dict]) -> list[dict]:
    region_easy = [r for r in rows if r.get("region") == "easy_to_learn"]
    candidates = region_easy or list(rows)
    return sorted(
        candidates,
        key=lambda r: (
            -float(r.get("confidence", 0.0)),
            float(r.get("variability", 0.0)),
            -float(r.get("correctness", 0.0)),
        ),
    )


def _select_flips(
    rows: list[dict],
    *,
    dataset: str,
    noise_ratio: float,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    if noise_ratio < 0.0 or noise_ratio > 1.0:
        raise ValueError("--noise-ratio must be in [0, 1]")
    if not rows:
        raise ValueError("cannot select label flips from an empty input")

    rng = random.Random(seed)
    candidates = _easy_candidates(rows)
    n_flip = max(1, int(len(rows) * noise_ratio)) if noise_ratio > 0 else 0
    n_flip = min(n_flip, len(candidates))
    selected = list(candidates[:n_flip])
    rng.shuffle(selected)

    flips: list[dict] = []
    overrides: dict[str, int] = {}
    for row in selected:
        if "gold_label" not in row:
            raise ValueError("input rows must contain gold_label to create label flips")
        guid = str(row["guid"])
        original = int(row["gold_label"])
        flipped = _flip_label(original, dataset, rng)
        overrides[guid] = flipped
        flips.append(
            {
                "guid": guid,
                "original_label": original,
                "new_label": flipped,
                "confidence": float(row.get("confidence", 0.0)),
                "variability": float(row.get("variability", 0.0)),
                "correctness": float(row.get("correctness", 0.0)),
                "region": row.get("region", "unknown"),
            }
        )
    return flips, overrides


def _write_train_command(
    path: Path,
    *,
    args: argparse.Namespace,
    overrides_path: Path,
    output_log: Path,
    summary_path: Path,
    metrics_path: Path,
    figures_dir: Path,
) -> None:
    cmd = [
        "python",
        "scripts/train_and_collect_dynamics.py",
        "--dataset",
        args.dataset,
        "--preset",
        args.preset,
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--max-length",
        str(args.max_length),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-eval-samples",
        str(args.max_eval_samples),
        "--label-overrides",
        str(overrides_path),
        "--output",
        str(output_log),
        "--summary-out",
        str(summary_path),
        "--metrics-out",
        str(metrics_path),
        "--figures-dir",
        str(figures_dir),
        "--subset-name",
        "label_flipped_easy_examples",
        "--subset-strategy",
        "noise_injection",
        "--seed",
        str(args.seed),
    ]
    if args.dataset == "winogrande":
        cmd.extend(["--winogrande-config", args.winogrande_config])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])
    if args.no_wandb:
        cmd.append("--no-wandb")
    if args.no_fp16:
        cmd.append("--no-fp16")
    if args.no_4bit:
        cmd.append("--no-4bit")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(" ".join(shlex.quote(part) for part in cmd))
        f.write("\n")
    path.chmod(0o755)


def _collect_dynamics_from_logs(log_path: Path) -> list[dict]:
    from ml_cartography.core.dynamics import (
        add_epoch_observation,
        build_record,
        summarize_record,
    )

    records = {}
    for row in read_jsonl(log_path):
        guid = str(row["guid"])
        gold = int(row.get("gold_label", 0))
        if guid not in records:
            records[guid] = build_record(guid, gold)
        add_epoch_observation(
            records[guid],
            float(row["prob_gold"]),
            int(row.get("pred_label", 0)),
        )
    return [summarize_record(r) for r in records.values()]


def _run_retraining(
    args: argparse.Namespace,
    overrides: dict[str, int],
    *,
    output_log: Path,
    summary_path: Path,
    metrics_path: Path,
    figures_dir: Path,
) -> dict:
    load_hf_credentials()
    from ml_cartography.training.glue_trainer import (
        MODEL_PRESETS,
        TrainConfig,
        apply_preset_defaults,
        train_and_collect_dynamics,
    )

    model_name = args.model_name or MODEL_PRESETS.get(args.preset, args.preset)
    cfg = TrainConfig(
        dataset=args.dataset,
        model_name=model_name,
        max_train_samples=None if args.max_train_samples == 0 else args.max_train_samples,
        max_eval_samples=None if args.max_eval_samples == 0 else args.max_eval_samples,
        epochs=args.epochs,
        batch_size=args.batch_size or 32,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        fp16=not args.no_fp16,
        output_logs=output_log,
        checkpoint_dir=args.output_dir / "checkpoints" if args.save_checkpoints else None,
        label_overrides=overrides,
        dynamic_snapshots=False,
        snapshot_dir=figures_dir,
        winogrande_config=args.winogrande_config,
    )
    cfg = apply_preset_defaults(cfg, args.preset)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.no_4bit:
        cfg.load_in_4bit = False

    wandb_run = None
    if use_wandb(args):
        import wandb

        wandb_run = wandb.run

    summary = train_and_collect_dynamics(cfg, metrics_log=metrics_path, wandb_run=wandb_run)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def _feature_matrix(rows: list[dict]):
    return [float(r["confidence"]) for r in rows]


def _balanced_split(
    rows: list[dict],
    noise_guids: set[str],
    *,
    seed: int,
    test_fraction: float,
) -> tuple[list[dict], list[int], list[dict], list[int]]:
    rng = random.Random(seed)
    noisy = [r for r in rows if str(r["guid"]) in noise_guids]
    clean = [r for r in rows if str(r["guid"]) not in noise_guids]
    if not noisy:
        raise ValueError("no noisy rows available for detector training")
    if len(clean) < len(noisy):
        raise ValueError("not enough clean rows to build a balanced detector dataset")

    rng.shuffle(noisy)
    rng.shuffle(clean)
    clean = clean[: len(noisy)]

    n_test = int(len(noisy) * test_fraction)
    if len(noisy) > 1:
        n_test = max(1, min(n_test, len(noisy) - 1))
    else:
        n_test = 0

    test_rows = noisy[:n_test] + clean[:n_test]
    test_y = [1] * n_test + [0] * n_test
    train_rows = noisy[n_test:] + clean[n_test:]
    train_y = [1] * (len(noisy) - n_test) + [0] * (len(clean) - n_test)

    paired = list(zip(train_rows, train_y))
    rng.shuffle(paired)
    train_rows = [r for r, _ in paired]
    train_y = [y for _, y in paired]

    paired = list(zip(test_rows, test_y))
    rng.shuffle(paired)
    test_rows = [r for r, _ in paired]
    test_y = [y for _, y in paired]
    return train_rows, train_y, test_rows, test_y


def _train_detector(
    retrained_rows: list[dict],
    original_rows: list[dict],
    noise_guids: set[str],
    *,
    seed: int,
    test_fraction: float,
) -> tuple[dict, list[dict], list[dict]]:
    train_rows, train_y, test_rows, test_y = _balanced_split(
        retrained_rows,
        noise_guids,
        seed=seed,
        test_fraction=test_fraction,
    )

    train_conf = _feature_matrix(train_rows)

    def prf(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    candidates = sorted(set(train_conf))
    if len(candidates) > 1:
        thresholds = [
            (candidates[i] + candidates[i + 1]) / 2.0 for i in range(len(candidates) - 1)
        ]
    else:
        thresholds = candidates

    best = {"threshold": thresholds[0], "direction": "le", "f1": -1.0}
    for threshold in thresholds:
        for direction in ("le", "ge"):
            pred = [
                int(c <= threshold) if direction == "le" else int(c >= threshold)
                for c in train_conf
            ]
            _precision, _recall, f1 = prf(train_y, pred)
            if f1 > best["f1"]:
                best = {"threshold": threshold, "direction": direction, "f1": f1}

    def predict_one(confidence: float) -> int:
        threshold = float(best["threshold"])
        if best["direction"] == "le":
            return int(confidence <= threshold)
        return int(confidence >= threshold)

    def score_one(confidence: float) -> float:
        threshold = float(best["threshold"])
        margin = threshold - confidence if best["direction"] == "le" else confidence - threshold
        return margin

    if test_rows:
        y_pred = [predict_one(c) for c in _feature_matrix(test_rows)]
        precision, recall, f1 = prf(test_y, y_pred)
    else:
        precision = recall = f1 = 0.0

    original_conf = _feature_matrix(original_rows)
    original_pred = [predict_one(c) for c in original_conf]
    original_score = [score_one(c) for c in original_conf]

    predictions: list[dict] = []
    flagged: list[dict] = []
    for row, pred, score in zip(original_rows, original_pred, original_score):
        out = {
            "guid": row["guid"],
            "predicted_noise": bool(int(pred)),
            "noise_score": float(score),
            "confidence": float(row["confidence"]),
            "variability": float(row["variability"]),
            "correctness": float(row.get("correctness", 0.0)),
            "region": row.get("region", "unknown"),
        }
        predictions.append(out)
        if int(pred) == 1:
            flagged.append(out)

    summary = {
        "detector_feature": "confidence",
        "balanced_train_rows": len(train_rows),
        "balanced_test_rows": len(test_rows),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "threshold": float(best["threshold"]),
        "threshold_direction": str(best["direction"]),
        "predicted_noise_count_original": len(flagged),
        "original_row_count": len(original_rows),
    }
    return summary, predictions, flagged


def _before_after_shift(
    original_rows: list[dict],
    retrained_rows: list[dict],
    noise_guids: set[str],
) -> tuple[dict, list[dict]]:
    original_by_guid = {str(r["guid"]): r for r in original_rows}
    shift_rows: list[dict] = []
    for row in retrained_rows:
        guid = str(row["guid"])
        before = original_by_guid.get(guid)
        if before is None:
            continue
        out = {
            "guid": guid,
            "injected_noise": guid in noise_guids,
            "confidence_before": float(before["confidence"]),
            "confidence_after": float(row["confidence"]),
            "variability_before": float(before["variability"]),
            "variability_after": float(row["variability"]),
            "correctness_before": float(before.get("correctness", 0.0)),
            "correctness_after": float(row.get("correctness", 0.0)),
            "region_before": before.get("region", "unknown"),
        }
        out["confidence_delta"] = out["confidence_after"] - out["confidence_before"]
        out["variability_delta"] = out["variability_after"] - out["variability_before"]
        shift_rows.append(out)

    noisy = [r for r in shift_rows if r["injected_noise"]]
    clean = [r for r in shift_rows if not r["injected_noise"]]

    def mean(rows: list[dict], key: str) -> float:
        return sum(float(r[key]) for r in rows) / len(rows) if rows else 0.0

    summary = {
        "noisy_confidence_before_mean": mean(noisy, "confidence_before"),
        "noisy_confidence_after_mean": mean(noisy, "confidence_after"),
        "noisy_variability_before_mean": mean(noisy, "variability_before"),
        "noisy_variability_after_mean": mean(noisy, "variability_after"),
        "clean_confidence_before_mean": mean(clean, "confidence_before"),
        "clean_confidence_after_mean": mean(clean, "confidence_after"),
        "clean_variability_before_mean": mean(clean, "variability_before"),
        "clean_variability_after_mean": mean(clean, "variability_after"),
    }
    return summary, shift_rows


def _human_eval_sample(
    predictions: list[dict],
    *,
    n_per_class: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    noisy = [r for r in predictions if r["predicted_noise"]]
    clean = [r for r in predictions if not r["predicted_noise"]]
    rng.shuffle(noisy)
    rng.shuffle(clean)
    sample = noisy[:n_per_class] + clean[:n_per_class]
    rng.shuffle(sample)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    parser.add_argument("--input", type=Path, default=None, help="Original coordinates-with-regions JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/noise_detection_paper"))
    parser.add_argument("--noise-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--human-eval-n", type=int, default=50)
    parser.add_argument("--train", action="store_true", help="Run noised retraining and detector.")
    parser.add_argument(
        "--retrained-coordinates",
        type=Path,
        default=None,
        help="Use existing coordinates from noised retraining instead of running --train.",
    )
    parser.add_argument("--dataset", choices=["snli", "mnli", "qnli", "winogrande"], default="winogrande")
    parser.add_argument("--preset", default="roberta-large")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--winogrande-config", default="winogrande_xl")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    input_path = args.input or Path(cfg["coordinates_with_regions_output"])
    original_rows = read_jsonl(input_path)
    if not original_rows:
        raise ValueError(f"no rows found in {input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    flips_path = args.output_dir / "flipped_examples.jsonl"
    overrides_path = args.output_dir / "label_overrides.json"
    output_log = args.output_dir / "noised_epoch_predictions.jsonl"
    summary_path = args.output_dir / "noised_training_summary.json"
    metrics_path = args.output_dir / "noised_training_metrics.jsonl"
    figures_dir = args.output_dir / "figures"
    command_path = args.output_dir / "retrain_noised.sh"
    retrained_coords_path = args.output_dir / "noised_cartography_coordinates.jsonl"
    shift_path = args.output_dir / "before_after_shift.jsonl"
    predictions_path = args.output_dir / "original_noise_predictions.jsonl"
    flagged_path = args.output_dir / "predicted_noisy_original.jsonl"
    human_eval_path = args.output_dir / "human_eval_sample.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    flips, overrides = _select_flips(
        original_rows,
        dataset=args.dataset,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
    )
    noise_guids = set(overrides)
    write_jsonl(flips_path, flips)
    write_json(overrides_path, {"label_overrides": overrides})
    _write_train_command(
        command_path,
        args=args,
        overrides_path=overrides_path,
        output_log=output_log,
        summary_path=summary_path,
        metrics_path=metrics_path,
        figures_dir=figures_dir,
    )

    init_wandb(
        args,
        job_type="paper_noise_detection",
        config={
            "input": str(input_path),
            "output_dir": str(args.output_dir),
            "noise_ratio": args.noise_ratio,
            "num_flipped": len(flips),
            "dataset": args.dataset,
            "preset": args.preset,
            "train": args.train,
        },
    )

    wandb_group = getattr(args, "wandb_group", None) or args.wandb_run_name or "fig4-noise"
    manifest = {
        "paper_task": "Noise injection and automatic noise detection (§5)",
        "tag": wandb_group,
        "wandb_group": wandb_group,
        "input": str(input_path),
        "output_dir": str(args.output_dir),
        "dataset": args.dataset,
        "preset": args.preset,
        "noise_ratio": args.noise_ratio,
        "seed": args.seed,
        "num_flipped": len(flips),
        "paths": {
            "flipped_examples": str(flips_path),
            "label_overrides": str(overrides_path),
            "retrain_command": str(command_path),
        },
    }

    train_summary: dict = {}
    if args.train:
        train_summary = _run_retraining(
            args,
            overrides,
            output_log=output_log,
            summary_path=summary_path,
            metrics_path=metrics_path,
            figures_dir=figures_dir,
        )
        retrained_rows = _collect_dynamics_from_logs(output_log)
        write_jsonl(retrained_coords_path, retrained_rows)
    elif args.retrained_coordinates:
        retrained_rows = read_jsonl(args.retrained_coordinates)
    else:
        retrained_rows = []

    if retrained_rows:
        shift_summary, shift_rows = _before_after_shift(original_rows, retrained_rows, noise_guids)
        detector_summary, predictions, flagged = _train_detector(
            retrained_rows,
            original_rows,
            noise_guids,
            seed=args.seed,
            test_fraction=args.test_fraction,
        )
        human_eval = _human_eval_sample(
            predictions,
            n_per_class=args.human_eval_n,
            seed=args.seed,
        )
        write_jsonl(shift_path, shift_rows)
        write_jsonl(predictions_path, predictions)
        write_jsonl(flagged_path, flagged)
        write_jsonl(human_eval_path, human_eval)

        manifest["train_summary"] = train_summary
        manifest["shift_summary"] = shift_summary
        manifest["detector_summary"] = detector_summary
        manifest["paths"].update(
            {
                "noised_epoch_predictions": str(output_log),
                "noised_coordinates": str(retrained_coords_path if args.train else args.retrained_coordinates),
                "before_after_shift": str(shift_path),
                "original_noise_predictions": str(predictions_path),
                "predicted_noisy_original": str(flagged_path),
                "human_eval_sample": str(human_eval_path),
            }
        )

        if use_wandb(args):
            import wandb

            wandb.log(
                {
                    "num_flipped": len(flips),
                    **{f"shift/{k}": v for k, v in shift_summary.items()},
                    **{f"detector/{k}": v for k, v in detector_summary.items()},
                }
            )
    write_json(manifest_path, manifest)

    print(f"flipped {len(flips)} examples -> {flips_path}")
    print(f"label overrides -> {overrides_path}")
    print(f"retrain command -> {command_path}")
    print(f"manifest -> {manifest_path}")
    if not args.train and not args.retrained_coordinates:
        print("next: run with --train, or execute retrain_noised.sh and pass --retrained-coordinates")
    elif retrained_rows:
        print(f"predicted noisy examples -> {flagged_path}")
        print(f"human eval sample -> {human_eval_path}")

    finish_wandb()


if __name__ == "__main__":
    main()
