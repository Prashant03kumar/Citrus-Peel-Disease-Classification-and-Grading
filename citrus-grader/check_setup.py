import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
MINIMUM_PYTHON = (3, 10)
REQUIRED_METRIC_FIELDS = {
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
IMPORTS = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("pillow", "PIL"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("python-multipart", "multipart"),
    ("pydantic", "pydantic"),
    ("pytest", "pytest"),
    ("httpx", "httpx"),
)
failures: list[str] = []


def report(label: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    if not passed:
        failures.append(label)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def check_python() -> None:
    current = sys.version_info
    passed = current[:2] >= MINIMUM_PYTHON
    report(
        "Python version",
        passed,
        f"{current.major}.{current.minor}.{current.micro}"
        + (" (requires 3.10 or newer)" if not passed else ""),
    )


def check_imports() -> None:
    for package, module in IMPORTS:
        try:
            importlib.import_module(module)
        except Exception as error:
            report(f"Import {package}", False, f"{type(error).__name__}: {error}")
        else:
            report(f"Import {package}", True)


def check_artifact(path: Path, label: str) -> bool:
    passed = path.is_file() and path.stat().st_size > 0
    report(label, passed, str(path.relative_to(PROJECT_ROOT)) if passed else "file missing or empty")
    return passed


def check_classes_and_samples() -> None:
    classes_path = PROJECT_ROOT / "models" / "classes.json"
    if not check_artifact(classes_path, "Class names artifact"):
        return
    try:
        class_names = load_json(classes_path)
    except (OSError, json.JSONDecodeError) as error:
        report("Class names JSON", False, str(error))
        return

    valid_classes = (
        isinstance(class_names, list)
        and bool(class_names)
        and all(isinstance(name, str) and name for name in class_names)
        and len(set(class_names)) == len(class_names)
    )
    report("Class names JSON", valid_classes, ", ".join(class_names) if valid_classes else "invalid class list")
    if not valid_classes:
        return

    samples_root = PROJECT_ROOT / "backend" / "static" / "samples"
    missing = []
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for class_name in class_names:
        class_dir = samples_root / class_name
        images = [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ] if class_dir.is_dir() else []
        if len(images) < 3:
            missing.append(f"{class_name} ({len(images)} images)")
    report(
        "Sample images for every class",
        not missing,
        "3 or more images per class" if not missing else "; ".join(missing),
    )


def check_metrics() -> None:
    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"
    if not check_artifact(metrics_path, "Metrics artifact"):
        return
    try:
        metrics = load_json(metrics_path)
    except (OSError, json.JSONDecodeError) as error:
        report("Metrics JSON schema", False, str(error))
        return

    if not isinstance(metrics, dict):
        report("Metrics JSON schema", False, "expected a JSON object")
        return
    missing_fields = REQUIRED_METRIC_FIELDS - metrics.keys()
    valid_accuracy = isinstance(metrics.get("accuracy"), (int, float)) and 0 <= metrics["accuracy"] <= 1
    valid_classes = isinstance(metrics.get("per_class"), dict) and bool(metrics["per_class"])
    report(
        "Metrics JSON schema",
        not missing_fields and valid_accuracy and valid_classes,
        "all required fields and accuracy present"
        if not missing_fields and valid_accuracy and valid_classes
        else f"missing={sorted(missing_fields)}, valid_accuracy={valid_accuracy}, valid_classes={valid_classes}",
    )


def check_node() -> None:
    node_path = shutil.which("node")
    if not node_path:
        report("Node.js installed", False, "node was not found on PATH")
        return
    try:
        completed = subprocess.run(
            [node_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        report("Node.js installed", False, str(error))
        return
    version = completed.stdout.strip()
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        report("Node.js installed", False, f"unrecognized version: {version}")
        return
    major, minor, _ = map(int, match.groups())
    supported = major > 22 or (major == 22 and minor >= 12) or (
        major == 20 and minor >= 19
    )
    report(
        "Node.js installed",
        supported,
        f"{version} (Vite 8 requires Node 20.19+ or 22.12+)",
    )


def check_npm() -> None:
    npm_path = shutil.which("npm")
    if not npm_path:
        report("npm installed", False, "npm was not found on PATH")
        return
    try:
        completed = subprocess.run(
            [npm_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        report("npm installed", False, str(error))
        return
    report("npm installed", True, completed.stdout.strip())


def main() -> int:
    print("Citrus Grader setup check")
    print("=" * 28)
    check_python()
    check_imports()
    check_artifact(
        PROJECT_ROOT / "models" / "citrus_effb0.pth",
        "Final model weights",
    )
    check_classes_and_samples()
    check_metrics()
    check_node()
    check_npm()
    node_modules = PROJECT_ROOT / "frontend" / "node_modules"
    report("Frontend dependencies", node_modules.is_dir(), "frontend/node_modules exists" if node_modules.is_dir() else "run npm install in frontend/")

    print("=" * 28)
    if failures:
        print(f"Setup check: FAIL ({len(failures)} check(s) failed)")
        return 1
    print("Setup check: PASS (all checks passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
