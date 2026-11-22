import argparse
import json
import os
from pathlib import Path
import time

import torch

if __package__:
    from .config import IMG_SIZE, MODELS_DIR
    from .evaluate import load_weights
    from .model import create_efficientnet_b0
    from .train import get_device
else:
    from config import IMG_SIZE, MODELS_DIR
    from evaluate import load_weights
    from model import create_efficientnet_b0
    from train import get_device


def benchmark_device(
    model_path: Path,
    num_classes: int,
    device: torch.device,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, float]:
    model = create_efficientnet_b0(num_classes, pretrained=False).to(device)
    load_weights(model, model_path, device)
    model.eval()
    image = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        durations = []
        for _ in range(timed_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)

    milliseconds_per_image = sum(durations) * 1000 / timed_runs
    return {
        "ms_per_image": milliseconds_per_image,
        "images_per_sec": 1000 / milliseconds_per_image,
    }


def benchmark_model(
    tag: str = "clahe",
    warmup_runs: int = 10,
    timed_runs: int = 100,
) -> dict[str, object]:
    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError("Warm-up count must be non-negative and timed runs must be positive.")
    classes = json.loads(
        (MODELS_DIR / "classes.json").read_text(encoding="utf-8")
    )
    model_path = MODELS_DIR / f"citrus_effb0_{tag}.pth"
    torch.set_num_threads(min(os.cpu_count() or 1, 8))

    cpu_result = benchmark_device(
        model_path,
        len(classes),
        torch.device("cpu"),
        warmup_runs,
        timed_runs,
    )
    result: dict[str, object] = {
        "cpu_ms_per_image": cpu_result["ms_per_image"],
        "cpu_images_per_sec": cpu_result["images_per_sec"],
        "gpu_ms_per_image": None,
        "gpu_images_per_sec": None,
    }
    if torch.cuda.is_available():
        gpu_result = benchmark_device(
            model_path,
            len(classes),
            torch.device("cuda"),
            warmup_runs,
            timed_runs,
        )
        result["gpu_ms_per_image"] = gpu_result["ms_per_image"]
        result["gpu_images_per_sec"] = gpu_result["images_per_sec"]

    print(
        f"CPU: {result['cpu_ms_per_image']:.3f} ms/image, "
        f"{result['cpu_images_per_sec']:.2f} images/sec"
    )
    if result["gpu_ms_per_image"] is None:
        print("GPU: unavailable")
    else:
        print(
            f"GPU: {result['gpu_ms_per_image']:.3f} ms/image, "
            f"{result['gpu_images_per_sec']:.2f} images/sec"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark saved-model inference speed.")
    parser.add_argument("--tag", default="clahe")
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--timed-runs", type=int, default=100)
    args = parser.parse_args()
    benchmark_model(args.tag, args.warmup_runs, args.timed_runs)


if __name__ == "__main__":
    main()
