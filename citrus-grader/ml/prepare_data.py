import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

if __package__:
    from .config import DATA_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR, SEED, VAL_FRACTION
else:
    from config import DATA_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR, SEED, VAL_FRACTION

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*") if is_image_file(path))


def split_is_empty(directory: Path) -> bool:
    return not image_files(directory)


def copy_split_from_source(source_dir: Path, split: str, destination: Path) -> None:
    source_split = source_dir / split
    if not source_split.is_dir():
        raise FileNotFoundError(
            f"Cannot populate data/{split}: source split does not exist: {source_split}"
        )
    source_files = [path for path in source_split.rglob("*") if path.is_file()]
    if not source_files:
        raise ValueError(f"Cannot populate data/{split}: source split is empty: {source_split}")

    for source_file in source_files:
        relative_path = source_file.relative_to(source_split)
        destination_file = destination / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        if not destination_file.exists():
            shutil.copy2(source_file, destination_file)
    print(f"Copied missing data/{split} files from {source_split} (original left untouched).")


def ensure_source_splits(source_dir: Optional[Path]) -> None:
    for split in ("train", "test"):
        split_dir = DATA_DIR / split
        if not split_is_empty(split_dir):
            continue
        split_dir.mkdir(parents=True, exist_ok=True)
        if source_dir is None:
            raise FileNotFoundError(
                f"data/{split} is missing or has no images. Supply the original dataset "
                "folder with --source-dir or CITRUS_SOURCE_DATA_DIR; no source files were changed."
            )
        copy_split_from_source(source_dir, split, split_dir)
        if split_is_empty(split_dir):
            raise ValueError(f"No images were copied into data/{split} from {source_dir}.")


def class_names(split: str) -> list[str]:
    split_dir = DATA_DIR / split
    if not split_dir.is_dir():
        return []
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


def validate_train_test_classes() -> list[str]:
    train_classes = class_names("train")
    test_classes = class_names("test")
    if train_classes != test_classes:
        raise ValueError(
            "Train/test class folders must match exactly. "
            f"Train: {train_classes}; test: {test_classes}."
        )
    if not train_classes:
        raise ValueError("No class folders found in data/train and data/test.")
    return train_classes


def find_and_remove_corrupt_images() -> list[str]:
    corrupt_paths = []
    for path in image_files(DATA_DIR):
        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError, SyntaxError) as error:
            relative_path = path.relative_to(PROJECT_ROOT).as_posix()
            corrupt_paths.append((relative_path, path, str(error)))

    for relative_path, path, _ in corrupt_paths:
        path.unlink()
        print(f"Removed unreadable image: {relative_path}")
    return [relative_path for relative_path, _, _ in corrupt_paths]


def list_non_image_files() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in DATA_DIR.rglob("*")
        if path.is_file() and not is_image_file(path)
    )


def create_validation_split(classes: list[str]) -> None:
    if image_files(DATA_DIR / "val"):
        print("Validation images already exist; skipping validation split.")
        return

    rng = random.Random(SEED)
    for class_name in classes:
        train_dir = DATA_DIR / "train" / class_name
        val_dir = DATA_DIR / "val" / class_name
        train_images = image_files(train_dir)
        if len(train_images) < 2:
            raise ValueError(
                f"Cannot create a train/validation split for {class_name}: "
                f"found only {len(train_images)} training images."
            )
        val_count = min(len(train_images) - 1, math.ceil(len(train_images) * VAL_FRACTION))
        selected = set(rng.sample(train_images, val_count))
        val_dir.mkdir(parents=True, exist_ok=True)
        for source in train_images:
            if source not in selected:
                continue
            destination = val_dir / source.name
            if destination.exists():
                raise FileExistsError(
                    f"Cannot move validation image because the destination exists: {destination}"
                )
            shutil.move(str(source), str(destination))
        print(f"Moved {val_count} {class_name} images from train to val.")


def validate_split_classes(classes: list[str]) -> None:
    for split in SPLITS:
        actual = class_names(split)
        if actual != classes:
            raise ValueError(
                f"Class folders in data/{split} must match train/test classes. "
                f"Expected {classes}; found {actual}."
            )


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_train_test_duplicates() -> list[dict[str, str]]:
    test_by_hash: dict[str, list[str]] = defaultdict(list)
    for path in image_files(DATA_DIR / "test"):
        test_by_hash[md5_file(path)].append(path.relative_to(PROJECT_ROOT).as_posix())

    duplicates = []
    for train_path in image_files(DATA_DIR / "train"):
        digest = md5_file(train_path)
        for test_path in test_by_hash.get(digest, []):
            duplicates.append(
                {
                    "train": train_path.relative_to(PROJECT_ROOT).as_posix(),
                    "test": test_path,
                    "md5": digest,
                }
            )
    return duplicates


def write_json_if_changed(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def write_sample_image(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = 600 / max(width, height)
        size = (round(width * scale), round(height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS)

        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    content = buffer.getvalue()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.read_bytes() != content:
        destination.write_bytes(content)


def count_images_by_class(classes: list[str]) -> dict[str, dict[str, int]]:
    counts = {}
    for class_name in classes:
        counts[class_name] = {
            split: len(image_files(DATA_DIR / split / class_name))
            for split in SPLITS
        }
    return counts


def print_report(
    classes: list[str],
    counts: dict[str, dict[str, int]],
    corrupt_removed: list[str],
    non_images: list[str],
    duplicates: list[dict[str, str]],
    imbalance_ratio: float,
) -> None:
    print(f"Classes found: {classes}")
    print("\nImage counts by class and split:")
    print(f"{'Class':<24}{'Train':>10}{'Val':>10}{'Test':>10}")
    print("-" * 54)
    for class_name in classes:
        class_counts = counts[class_name]
        print(
            f"{class_name:<24}{class_counts['train']:>10}"
            f"{class_counts['val']:>10}{class_counts['test']:>10}"
        )
    print("-" * 54)
    print(
        f"{'TOTAL':<24}"
        f"{sum(counts[name]['train'] for name in classes):>10}"
        f"{sum(counts[name]['val'] for name in classes):>10}"
        f"{sum(counts[name]['test'] for name in classes):>10}"
    )
    print(f"\nCorrupt/unreadable images removed: {len(corrupt_removed)}")
    for path in corrupt_removed:
        print(f"  {path}")
    print(f"Non-image files found (left in place): {len(non_images)}")
    for path in non_images:
        print(f"  {path}")
    print(f"Exact train/test duplicate pairs (MD5): {len(duplicates)}")
    print(f"Training class imbalance ratio: {imbalance_ratio:.3f}")

    for class_name in classes:
        if counts[class_name]["train"] < 100:
            print(
                f"WARNING: {class_name} has fewer than 100 training images "
                f"({counts[class_name]['train']})."
            )
    if len(classes) > 1:
        training_counts = [counts[name]["train"] for name in classes]
        if min(training_counts) and max(training_counts) > 2 * min(training_counts):
            print(
                "WARNING: largest training class has more than twice the images "
                "of the smallest class."
            )


def prepare_data(source_dir: Optional[Path] = None) -> dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_source_splits(source_dir)

    classes = validate_train_test_classes()
    corrupt_removed = find_and_remove_corrupt_images()
    non_images = list_non_image_files()
    create_validation_split(classes)
    validate_split_classes(classes)

    class_counts = count_images_by_class(classes)
    training_counts = [class_counts[name]["train"] for name in classes]
    if not training_counts:
        raise ValueError("No training classes available for dataset statistics.")
    smallest = min(training_counts)
    imbalance_ratio = max(training_counts) / smallest if smallest else 0.0
    duplicates = find_train_test_duplicates()

    stats = {
        "classes": classes,
        "train": sum(class_counts[name]["train"] for name in classes),
        "val": sum(class_counts[name]["val"] for name in classes),
        "test": sum(class_counts[name]["test"] for name in classes),
        "per_class": class_counts,
        "imbalance_ratio": round(imbalance_ratio, 6),
        "duplicates_train_test": len(duplicates),
        "corrupt_removed": len(corrupt_removed),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_if_changed(MODELS_DIR / "classes.json", classes)
    write_json_if_changed(REPORTS_DIR / "dataset_stats.json", stats)
    write_json_if_changed(
        REPORTS_DIR / "duplicates_train_test.json",
        {"duplicate_pairs": duplicates},
    )

    for class_name in classes:
        samples = image_files(DATA_DIR / "train" / class_name)[:3]
        if len(samples) != 3:
            raise ValueError(
                f"Cannot create 3 sample images for {class_name}; found {len(samples)}."
            )
        for index, source in enumerate(samples, start=1):
            write_sample_image(
                source,
                PROJECT_ROOT / "backend" / "static" / "samples" / class_name / f"{index}.jpg",
            )

    print_report(classes, class_counts, corrupt_removed, non_images, duplicates, imbalance_ratio)
    print(f"\nWrote {REPORTS_DIR / 'dataset_stats.json'}")
    print(f"Wrote {MODELS_DIR / 'classes.json'}")
    print(f"Wrote {REPORTS_DIR / 'duplicates_train_test.json'}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate citrus images, create a seeded validation split, and write stats."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Original dataset folder containing train/ and test/. Used only to copy a "
            "missing/empty project split; the source is never modified."
        ),
    )
    args = parser.parse_args()
    source_dir = args.source_dir
    if source_dir is None:
        environment_source = os.environ.get("CITRUS_SOURCE_DATA_DIR")
        if environment_source:
            source_dir = Path(environment_source)
    prepare_data(source_dir)


if __name__ == "__main__":
    main()
