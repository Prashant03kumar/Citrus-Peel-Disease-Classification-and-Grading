import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__:
    from .config import MODELS_DIR, REPORTS_DIR
else:
    from config import MODELS_DIR, REPORTS_DIR


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_metrics(metrics: dict[str, Any]) -> None:
    top_level_keys = {
        "model",
        "preprocessing",
        "epochs",
        "trained_at",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "per_class",
        "confusion_matrix",
        "dataset",
        "speed",
        "ablation",
    }
    require(set(metrics) == top_level_keys, "metrics.json has unexpected/missing top-level keys.")
    require(metrics["model"] == "efficientnet_b0", "model must be efficientnet_b0.")
    require(metrics["preprocessing"] == "clahe", "preprocessing must be clahe.")
    require(type(metrics["epochs"]) is int and metrics["epochs"] > 0, "epochs must be a positive int.")
    require(isinstance(metrics["trained_at"], str), "trained_at must be an ISO date string.")
    try:
        datetime.fromisoformat(metrics["trained_at"])
    except ValueError as error:
        raise ValueError("trained_at must be a valid ISO-8601 timestamp.") from error

    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        require(type(metrics[key]) in (int, float), f"{key} must be numeric.")

    classes = json.loads((MODELS_DIR / "classes.json").read_text(encoding="utf-8"))
    require(set(metrics["per_class"]) == set(classes), "per_class keys must match classes.json.")
    for class_name, values in metrics["per_class"].items():
        require(
            set(values) == {"precision", "recall", "f1", "support"},
            f"Invalid per_class schema for {class_name}.",
        )
        for key in ("precision", "recall", "f1"):
            require(type(values[key]) in (int, float), f"{class_name}.{key} must be numeric.")
        require(type(values["support"]) is int, f"{class_name}.support must be int.")

    confusion = metrics["confusion_matrix"]
    require(
        set(confusion) == {"labels", "matrix"}
        and confusion["labels"] == classes
        and len(confusion["matrix"]) == len(classes),
        "Invalid confusion_matrix schema or class ordering.",
    )
    require(
        all(
            len(row) == len(classes) and all(type(value) is int for value in row)
            for row in confusion["matrix"]
        ),
        "confusion_matrix.matrix must be a square integer matrix.",
    )

    dataset = metrics["dataset"]
    require(set(dataset) == {"train", "val", "test", "per_class"}, "Invalid dataset schema.")
    for split in ("train", "val", "test"):
        require(type(dataset[split]) is int, f"dataset.{split} must be int.")
    require(set(dataset["per_class"]) == set(classes), "dataset.per_class classes mismatch.")
    for class_name, counts in dataset["per_class"].items():
        require(set(counts) == {"train", "val", "test"}, f"Invalid counts for {class_name}.")
        require(all(type(counts[split]) is int for split in counts), "Dataset counts must be int.")

    speed = metrics["speed"]
    speed_keys = {
        "cpu_ms_per_image",
        "cpu_images_per_sec",
        "gpu_ms_per_image",
        "gpu_images_per_sec",
    }
    require(set(speed) == speed_keys, "Invalid speed schema.")
    for key in ("cpu_ms_per_image", "cpu_images_per_sec"):
        require(type(speed[key]) in (int, float), f"{key} must be numeric.")
    for key in ("gpu_ms_per_image", "gpu_images_per_sec"):
        require(speed[key] is None or type(speed[key]) in (int, float), f"{key} must be numeric or null.")

    ablation = metrics["ablation"]
    require(
        ablation is None
        or (
            set(ablation) == {"baseline_accuracy", "clahe_accuracy"}
            and all(type(value) in (int, float) for value in ablation.values())
        ),
        "ablation must be null or contain baseline_accuracy and clahe_accuracy.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate reports/metrics.json against the project contract.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=REPORTS_DIR / "metrics.json",
    )
    args = parser.parse_args()
    metrics = json.loads(args.path.read_text(encoding="utf-8"))
    validate_metrics(metrics)
    print(f"PASS: {args.path} matches the project metrics schema.")


if __name__ == "__main__":
    main()
