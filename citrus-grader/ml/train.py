import argparse
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

if __package__:
    from .clahe import CLAHETransform
    from .config import DATA_DIR, IMG_SIZE, MODELS_DIR, SEED
    from .model import create_efficientnet_b0
else:
    from clahe import CLAHETransform
    from config import DATA_DIR, IMG_SIZE, MODELS_DIR, SEED
    from model import create_efficientnet_b0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_transforms(use_clahe: bool) -> tuple[transforms.Compose, transforms.Compose]:
    clahe = [CLAHETransform()] if use_clahe else []
    train_transform = transforms.Compose(
        clahe
        + [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        clahe
        + [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def stratified_subset_indices(targets: list[int], limit: int, seed: int = SEED) -> list[int]:
    if limit >= len(targets):
        return list(range(len(targets)))
    if limit <= 0:
        raise ValueError("The training subset size must be greater than zero.")

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        grouped[target].append(index)
    if limit < len(grouped):
        raise ValueError(
            f"Training subset size {limit} is smaller than the {len(grouped)} classes."
        )

    exact_quotas = {
        target: limit * len(indices) / len(targets)
        for target, indices in grouped.items()
    }
    quotas = {
        target: min(len(grouped[target]), math.floor(quota))
        for target, quota in exact_quotas.items()
    }
    remaining = limit - sum(quotas.values())
    remainder_order = sorted(
        grouped,
        key=lambda target: (exact_quotas[target] - math.floor(exact_quotas[target]), target),
        reverse=True,
    )
    while remaining:
        assigned = False
        for target in remainder_order:
            if quotas[target] < len(grouped[target]):
                quotas[target] += 1
                remaining -= 1
                assigned = True
                if remaining == 0:
                    break
        if not assigned:
            raise ValueError("Could not allocate the requested stratified training subset.")

    rng = random.Random(seed)
    selected = []
    for target, indices in grouped.items():
        selected.extend(rng.sample(indices, quotas[target]))
    return sorted(selected)


def build_loaders(
    batch_size: int,
    use_clahe: bool,
    train_subset: Optional[int] = None,
) -> tuple[DataLoader, DataLoader, list[str], list[int]]:
    train_transform, evaluation_transform = make_transforms(use_clahe)
    train_data = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_data = datasets.ImageFolder(DATA_DIR / "val", transform=evaluation_transform)
    if train_data.classes != val_data.classes:
        raise ValueError(
            f"Train/validation classes differ: {train_data.classes} vs {val_data.classes}."
        )
    if not train_data.classes:
        raise ValueError("No class folders found in data/train.")

    train_indices = list(range(len(train_data)))
    if train_subset is not None:
        train_indices = stratified_subset_indices(train_data.targets, train_subset)
        train_data_for_loader: Dataset = Subset(train_data, train_indices)
    else:
        train_data_for_loader = train_data

    class_counts = Counter(train_data.targets[index] for index in train_indices)
    if any(class_counts[index] == 0 for index in range(len(train_data.classes))):
        raise ValueError("The training data must contain at least one image per class.")

    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = DataLoader(
        train_data_for_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    weights = [
        len(train_indices) / (len(train_data.classes) * class_counts[index])
        for index in range(len(train_data.classes))
    ]
    return train_loader, val_loader, train_data.classes, weights


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    freeze_backbone: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    if training and freeze_backbone:
        model.features.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

    if total == 0:
        raise ValueError("Cannot run an epoch with an empty data loader.")
    return total_loss / total, correct / total


def save_state_dict(model: nn.Module, path: Path) -> None:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state, path)


def train_model(
    use_clahe: bool = True,
    epochs_head: int = 3,
    epochs_finetune: int = 10,
    batch_size: int = 16,
    tag: Optional[str] = None,
    train_subset: Optional[int] = None,
) -> dict[str, object]:
    if epochs_head < 0 or epochs_finetune < 1:
        raise ValueError("Head epochs must be non-negative and finetune epochs at least one.")
    if batch_size < 1:
        raise ValueError("Batch size must be greater than zero.")
    resolved_tag = tag or ("clahe" if use_clahe else "baseline")
    if not resolved_tag or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in resolved_tag):
        raise ValueError("Tag may contain only letters, numbers, hyphens, and underscores.")

    seed_everything()
    thread_count = min(os.cpu_count() or 1, 8)
    torch.set_num_threads(thread_count)
    device = get_device()
    train_loader, val_loader, classes, class_weights = build_loaders(
        batch_size,
        use_clahe,
        train_subset,
    )
    model = create_efficientnet_b0(len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    model_path = MODELS_DIR / f"citrus_effb0_{resolved_tag}.pth"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = -1.0
    best_epoch = 0
    completed_epochs = 0

    def train_phase(
        phase_name: str,
        epochs: int,
        learning_rate: float,
        freeze_backbone: bool,
        early_stopping_patience: Optional[int] = None,
    ) -> None:
        nonlocal best_val_accuracy, best_epoch, completed_epochs
        if epochs == 0:
            return
        for parameter in model.parameters():
            parameter.requires_grad = not freeze_backbone
        if freeze_backbone:
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
        )
        epochs_without_improvement = 0
        for phase_epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_accuracy = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                freeze_backbone=freeze_backbone,
            )
            _, val_accuracy = run_epoch(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.perf_counter() - epoch_start
            completed_epochs += 1
            print(
                f"{phase_name} epoch {phase_epoch}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
                f"val_acc={val_accuracy:.4f}, epoch_time={elapsed:.1f}s, "
                f"estimated_time_per_epoch={elapsed:.1f}s",
                flush=True,
            )
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = completed_epochs
                save_state_dict(model, model_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    f"Early stopping {phase_name} after "
                    f"{epochs_without_improvement} epochs without validation improvement.",
                    flush=True,
                )
                break

    print(
        f"Training {resolved_tag}: classes={classes}, device={device}, "
        f"train_images={len(train_loader.dataset)}, val_images={len(val_loader.dataset)}, "
        f"test_images=not loaded",
        flush=True,
    )
    train_phase("Phase A (head)", epochs_head, 1e-3, freeze_backbone=True)
    train_phase(
        "Phase B (finetune)",
        epochs_finetune,
        1e-4,
        freeze_backbone=False,
        early_stopping_patience=4,
    )
    if best_epoch == 0:
        raise RuntimeError("Training completed without saving a best-validation checkpoint.")

    if resolved_tag == "clahe":
        shutil.copy2(model_path, MODELS_DIR / "citrus_effb0.pth")
    print(
        f"Best validation accuracy={best_val_accuracy:.4f} at epoch {best_epoch}; "
        f"saved {model_path}",
        flush=True,
    )
    return {
        "tag": resolved_tag,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "epochs_completed": completed_epochs,
        "model_path": model_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on citrus peel images.")
    clahe_options = parser.add_mutually_exclusive_group()
    clahe_options.add_argument("--clahe", dest="clahe", action="store_true")
    clahe_options.add_argument("--no-clahe", dest="clahe", action="store_false")
    parser.set_defaults(clahe=True)
    parser.add_argument("--epochs-head", type=int, default=3)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--tag",
        default=None,
        help="Checkpoint name (defaults to 'clahe' with CLAHE or 'baseline' without it).",
    )
    args = parser.parse_args()
    train_model(
        use_clahe=args.clahe,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
