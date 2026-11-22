import argparse
import json
from datetime import datetime, timezone

if __package__:
    from .benchmark import benchmark_model
    from .config import MODELS_DIR, REPORTS_DIR
    from .evaluate import evaluate_model
    from .train import train_model
    from .validate_metrics import validate_metrics
else:
    from benchmark import benchmark_model
    from config import MODELS_DIR, REPORTS_DIR
    from evaluate import evaluate_model
    from train import train_model
    from validate_metrics import validate_metrics


def write_metrics(
    clahe_training: dict[str, object],
    baseline_evaluation: dict[str, object],
    clahe_evaluation: dict[str, object],
    speed: dict[str, object],
) -> dict[str, object]:
    stats = json.loads((REPORTS_DIR / "dataset_stats.json").read_text(encoding="utf-8"))
    classes = json.loads((MODELS_DIR / "classes.json").read_text(encoding="utf-8"))
    dataset = {
        "train": stats["train"],
        "val": stats["val"],
        "test": stats["test"],
        "per_class": stats["per_class"],
    }
    metrics = {
        "model": "efficientnet_b0",
        "preprocessing": "clahe",
        "epochs": int(clahe_training["epochs_completed"]),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accuracy": clahe_evaluation["accuracy"],
        "macro_precision": clahe_evaluation["macro_precision"],
        "macro_recall": clahe_evaluation["macro_recall"],
        "macro_f1": clahe_evaluation["macro_f1"],
        "per_class": clahe_evaluation["per_class"],
        "confusion_matrix": clahe_evaluation["confusion_matrix"],
        "dataset": dataset,
        "speed": speed,
        "ablation": {
            "baseline_accuracy": baseline_evaluation["accuracy"],
            "clahe_accuracy": clahe_evaluation["accuracy"],
        },
    }
    if metrics["confusion_matrix"]["labels"] != classes:
        raise ValueError("Evaluated confusion-matrix labels differ from models/classes.json.")
    validate_metrics(metrics)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_metrics(metrics)
    print(f"Wrote {REPORTS_DIR / 'metrics.json'}")
    return metrics


def run(quick: bool = False) -> dict[str, object]:
    epochs_head = 1 if quick else 3
    epochs_finetune = 1 if quick else 10
    train_subset = 200 if quick else None

    print(
        f"Pipeline mode: {'quick' if quick else 'full'}; "
        f"head_epochs={epochs_head}, finetune_epochs={epochs_finetune}, "
        f"train_subset={train_subset or 'all'}"
    )
    print("\n=== Train baseline (without CLAHE) ===")
    train_model(
        use_clahe=False,
        epochs_head=epochs_head,
        epochs_finetune=epochs_finetune,
        batch_size=16,
        tag="baseline",
        train_subset=train_subset,
    )
    print("\n=== Evaluate baseline on test ===")
    baseline_evaluation = evaluate_model("baseline")

    print("\n=== Train CLAHE model ===")
    clahe_training = train_model(
        use_clahe=True,
        epochs_head=epochs_head,
        epochs_finetune=epochs_finetune,
        batch_size=16,
        tag="clahe",
        train_subset=train_subset,
    )
    print("\n=== Evaluate CLAHE model on test ===")
    clahe_evaluation = evaluate_model("clahe")

    print("\n=== Benchmark CLAHE model (10 warm-ups, 100 timed inferences) ===")
    speed = benchmark_model("clahe", warmup_runs=10, timed_runs=100)
    return write_metrics(
        clahe_training,
        baseline_evaluation,
        clahe_evaluation,
        speed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, benchmark, and summarize baseline and CLAHE models."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one epoch per phase and a stratified subset of 200 training images.",
    )
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
