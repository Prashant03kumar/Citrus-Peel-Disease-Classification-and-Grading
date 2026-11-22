from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    classes: list[str]


class ClassResponse(BaseModel):
    name: str
    display_name: str
    description: str
    sample_images: list[str]


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    grade: str
    decision: str
    inference_ms: float


class PerClassMetricsResponse(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class ConfusionMatrixResponse(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class DatasetClassCountsResponse(BaseModel):
    train: int
    val: int
    test: int


class DatasetMetricsResponse(BaseModel):
    train: int
    val: int
    test: int
    per_class: dict[str, DatasetClassCountsResponse]


class SpeedMetricsResponse(BaseModel):
    cpu_ms_per_image: float
    cpu_images_per_sec: float
    gpu_ms_per_image: float | None
    gpu_images_per_sec: float | None


class AblationMetricsResponse(BaseModel):
    baseline_accuracy: float
    clahe_accuracy: float


class MetricsResponse(BaseModel):
    model: str
    preprocessing: str
    epochs: int
    trained_at: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, PerClassMetricsResponse]
    confusion_matrix: ConfusionMatrixResponse
    dataset: DatasetMetricsResponse
    speed: SpeedMetricsResponse
    ablation: AblationMetricsResponse | None
