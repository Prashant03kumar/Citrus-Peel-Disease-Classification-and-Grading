import json
import sys
import time
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.clahe import CLAHETransform
from ml.model import create_efficientnet_b0

MODEL_PATH = PROJECT_ROOT / "models" / "citrus_effb0.pth"
CLASSES_PATH = PROJECT_ROOT / "models" / "classes.json"


class InferenceEngine:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        classes_path: Path = CLASSES_PATH,
    ) -> None:
        with classes_path.open(encoding="utf-8") as classes_file:
            classes = json.load(classes_file)
        if not isinstance(classes, list) or not classes or not all(
            isinstance(name, str) for name in classes
        ):
            raise ValueError(f"Expected a non-empty list of class names in {classes_path}.")

        self.classes: list[str] = classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_efficientnet_b0(len(self.classes), pretrained=False)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                CLAHETransform(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def predict(self, image_bytes: bytes) -> dict[str, object]:
        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - start) * 1000

        probabilities = torch.softmax(logits[0], dim=0).cpu().tolist()
        predicted_index = int(torch.argmax(logits[0]).item())
        return {
            "predicted_class": self.classes[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                class_name: float(probability)
                for class_name, probability in zip(self.classes, probabilities)
            },
            "inference_ms": inference_ms,
        }
