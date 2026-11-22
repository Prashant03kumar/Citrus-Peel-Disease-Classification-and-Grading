import json
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .grading import grade_prediction
from .inference import InferenceEngine, PROJECT_ROOT
from .schemas import ClassResponse, HealthResponse, MetricsResponse, PredictionResponse

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STATIC_DIR = PROJECT_ROOT / "backend" / "static"
SAMPLES_DIR = STATIC_DIR / "samples"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.inference = InferenceEngine()
    yield
    app.state.inference = None


app = FastAPI(
    title="Citrus Grader API",
    description=(
        "API for citrus peel disease classification and fruit grading. The current "
        "dataset contains only black-spot and citrus-canker, so outcomes are Grade B, "
        "Reject, or Review; Grade A becomes possible if a healthy class is added."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    engine: InferenceEngine | None = getattr(app.state, "inference", None)
    return HealthResponse(
        status="ok",
        model_loaded=engine is not None,
        classes=engine.classes if engine is not None else [],
    )


@app.get("/api/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    if not METRICS_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Metrics are unavailable: {METRICS_PATH.name} was not found.",
        )
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return MetricsResponse(**json.load(metrics_file))


def describe_class(class_name: str) -> str:
    normalized_name = class_name.lower().replace("_", "-")
    if "canker" in normalized_name:
        return "Canker causes raised, corky lesions on the peel, often surrounded by a yellow halo."
    if "black-spot" in normalized_name or "black spot" in normalized_name:
        return "Black spot appears as dark, sunken spots on the citrus peel."
    if "scab" in normalized_name:
        return "Scab appears as raised, corky bumps on the citrus peel."
    if "healthy" in normalized_name:
        return "Healthy citrus has clean, smooth peel without visible disease lesions."
    return "This class identifies a citrus peel condition not covered by the known categories."


@app.get("/api/classes", response_model=list[ClassResponse])
def classes() -> list[ClassResponse]:
    engine: InferenceEngine | None = getattr(app.state, "inference", None)
    if engine is None:
        return []
    class_responses = []
    for class_name in engine.classes:
        sample_dir = SAMPLES_DIR / class_name
        sample_urls = [
            f"/static/samples/{class_name}/{sample.name}"
            for sample in sorted(sample_dir.iterdir())
            if sample.is_file()
        ] if sample_dir.is_dir() else []
        class_responses.append(
            ClassResponse(
                name=class_name,
                display_name=" ".join(
                    word.capitalize() for word in class_name.replace("_", "-").split("-")
                ),
                description=describe_class(class_name),
                sample_images=sample_urls,
            )
        )
    return class_responses


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.lower().startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must have an image content type.")

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image files must be 10 MB or smaller.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                rgb_image = image.convert("RGB")
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        UnidentifiedImageError,
        ValueError,
    ) as error:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from error

    engine: InferenceEngine | None = getattr(app.state, "inference", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="The model is not loaded.")

    converted_bytes = BytesIO()
    rgb_image.save(converted_bytes, format="PNG")
    prediction = engine.predict(converted_bytes.getvalue())
    grade = grade_prediction(
        str(prediction["predicted_class"]),
        float(prediction["confidence"]),
    )
    return PredictionResponse(
        filename=file.filename or "",
        **prediction,
        **grade,
    )
