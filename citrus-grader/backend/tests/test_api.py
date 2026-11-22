import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.grading import grade_prediction
from backend.app.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PREDICTION_FIELDS = {
    "filename",
    "predicted_class",
    "confidence",
    "probabilities",
    "grade",
    "decision",
    "inference_ms",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model_and_classes(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "classes": json.loads((PROJECT_ROOT / "models" / "classes.json").read_text()),
    }


def test_metrics_contains_required_fields(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    assert {
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
    } <= response.json().keys()


def test_classes_include_resolvable_sample_images(client):
    response = client.get("/api/classes")
    assert response.status_code == 200
    classes = response.json()
    expected_names = json.loads((PROJECT_ROOT / "models" / "classes.json").read_text())
    assert [item["name"] for item in classes] == expected_names
    for class_info in classes:
        assert {
            "name",
            "display_name",
            "description",
            "sample_images",
        } == class_info.keys()
        assert len(class_info["sample_images"]) == 3
        for sample_url in class_info["sample_images"]:
            sample_response = client.get(sample_url)
            assert sample_response.status_code == 200


@pytest.mark.parametrize(
    "class_name",
    json.loads((PROJECT_ROOT / "models" / "classes.json").read_text()),
)
def test_predicts_real_test_image_with_contract_fields(client, class_name):
    image_path = sorted((PROJECT_ROOT / "data" / "test" / class_name).glob("*"))[0]
    with image_path.open("rb") as image_file:
        response = client.post(
            "/api/predict",
            files={"file": (image_path.name, image_file, "image/jpeg")},
        )

    assert response.status_code == 200
    result = response.json()
    assert result.keys() == EXPECTED_PREDICTION_FIELDS
    assert result["filename"] == image_path.name
    assert result["predicted_class"] in result["probabilities"]
    assert set(result["probabilities"]) == set(
        json.loads((PROJECT_ROOT / "models" / "classes.json").read_text())
    )
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert result["confidence"] == pytest.approx(
        result["probabilities"][result["predicted_class"]]
    )
    assert isinstance(result["inference_ms"], float)


def test_rejects_non_image_content_type(client):
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Upload must have an image content type."}


def test_rejects_corrupt_image(client):
    response = client.post(
        "/api/predict",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is not a valid image."}


def test_rejects_oversized_upload(client):
    response = client.post(
        "/api/predict",
        files={"file": ("large.jpg", b"x" * (10 * 1024 * 1024 + 1), "image/jpeg")},
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "Image files must be 10 MB or smaller."}


def test_grading_rules_and_confidence_threshold():
    assert grade_prediction("black-spot", 0.90) == {
        "grade": "B",
        "decision": "GRADE B (processing / juice)",
    }
    assert grade_prediction("citrus-canker", 0.95) == {
        "grade": "Reject",
        "decision": "REJECT (quarantine disease)",
    }
    assert grade_prediction("healthy", 0.95) == {
        "grade": "A",
        "decision": "GRADE A (fresh market)",
    }
    assert grade_prediction("scab", 0.95)["grade"] == "B"
    assert grade_prediction("unlisted-class", 0.99) == {
        "grade": "Review",
        "decision": "MANUAL INSPECTION",
    }
    assert grade_prediction("black-spot", 0.89) == {
        "grade": "Review",
        "decision": "MANUAL INSPECTION",
    }
