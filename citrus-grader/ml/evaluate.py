import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if __package__:
    from .clahe import CLAHETransform
    from .config import DATA_DIR, IMG_SIZE, MODELS_DIR, REPORTS_DIR, SEED
    from .model import create_efficientnet_b0
    from .train import IMAGENET_MEAN, IMAGENET_STD, get_device
else:
    from clahe import CLAHETransform
    from config import DATA_DIR, IMG_SIZE, MODELS_DIR, REPORTS_DIR, SEED
    from model import create_efficientnet_b0
    from train import IMAGENET_MEAN, IMAGENET_STD, get_device


def load_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def plot_confusion_matrix(
    matrix: np.ndarray,
    classes: list[str],
    path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=12,
            )
    axis.set_ylim(len(classes) - 0.5, -0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_model(tag: str = "clahe", batch_size: int = 16) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("Batch size must be greater than zero.")
    classes_path = MODELS_DIR / "classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    if not isinstance(classes, list) or not all(isinstance(name, str) for name in classes):
        raise ValueError(f"Invalid class list in {classes_path}.")

    use_clahe = tag == "clahe"
    image_transform = []
    if use_clahe:
        image_transform.append(CLAHETransform())
    image_transform.extend(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_data = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=transforms.Compose(image_transform),
    )
    if test_data.classes != classes:
        raise ValueError(
            f"Test classes {test_data.classes} do not match models/classes.json {classes}."
        )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    device = get_device()
    model = create_efficientnet_b0(len(classes), pretrained=False).to(device)
    weights_path = MODELS_DIR / f"citrus_effb0_{tag}.pth"
    load_weights(model, weights_path, device)
    model.eval()

    actual = []
    predicted = []
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            outputs = model(images)
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())
            actual.extend(labels.tolist())

    label_indices = list(range(len(classes)))
    matrix = confusion_matrix(actual, predicted, labels=label_indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=label_indices,
        zero_division=0,
    )
    result = {
        "accuracy": float(np.mean(np.asarray(actual) == np.asarray(predicted))),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(classes)
        },
        "confusion_matrix": {
            "labels": classes,
            "matrix": matrix.astype(int).tolist(),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        matrix,
        classes,
        REPORTS_DIR / f"confusion_matrix_{tag}.png",
        f"Confusion Matrix ({tag})",
    )
    if tag == "clahe":
        plot_confusion_matrix(
            matrix,
            classes,
            REPORTS_DIR / "confusion_matrix.png",
            "Confusion Matrix (CLAHE)",
        )

    print(f"\nTest results ({tag}; {len(test_data)} images):")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro precision: {result['macro_precision']:.4f}")
    print(f"Macro recall: {result['macro_recall']:.4f}")
    print(f"Macro F1: {result['macro_f1']:.4f}")
    for class_name, class_metrics in result["per_class"].items():
        print(
            f"  {class_name}: precision={class_metrics['precision']:.4f}, "
            f"recall={class_metrics['recall']:.4f}, f1={class_metrics['f1']:.4f}, "
            f"support={class_metrics['support']}"
        )
    print(f"Confusion matrix:\n{matrix}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on data/test.")
    parser.add_argument("--tag", default="clahe")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate_model(args.tag, args.batch_size)


if __name__ == "__main__":
    main()
