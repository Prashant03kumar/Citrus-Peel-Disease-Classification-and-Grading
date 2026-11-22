# Project purpose

`citrus-grader` classifies citrus peel diseases from fruit photos and grades fruit for sorting.
It uses a CPU-friendly PyTorch EfficientNet-B0 model with CLAHE preprocessing and serves predictions through a FastAPI backend and React dashboard.

# Folder structure

```text
citrus-grader/
  data/
    train/
    val/
    test/
  ml/
  models/
  reports/
  backend/
    app/
    static/
      samples/
    requirements.txt
  frontend/
  PROJECT_CONTEXT.md
  PROGRESS.md
  README.md
```

Phase 2 writes dataset preparation and preprocessing code in `ml/` and dataset counts to `reports/dataset_stats.json`. Phase 3 writes training and evaluation code in `ml/`, best weights and class ordering to `models/`, and metrics and the confusion-matrix image to `reports/`. Phase 4 writes the API in `backend/app/`, grading settings to `backend/grading_config.json`, and class sample photos to `backend/static/samples/`. Phase 5 writes the dashboard in `frontend/`. Phase 6 writes integration and validation updates to their relevant existing folders and project documentation.

# Data layout

Use `data/train/<class>/`, `data/val/<class>/`, and `data/test/<class>/`. Class names are discovered from folder names, never hardcoded in ML code. `models/classes.json` records class names in model index order.

# File contract

- `models/citrus_effb0.pth`: best model weights as a PyTorch `state_dict`.
- `models/classes.json`: list of class names in index order.
- `reports/dataset_stats.json`: dataset counts, including per-class counts.
- `reports/metrics.json`: evaluation and runtime metrics matching the schema below.
- `reports/confusion_matrix.png`: confusion matrix visualization.
- `backend/static/samples/<class>/*.jpg`: 3 sample images per class, named `1.jpg`, `2.jpg`, and `3.jpg`.
- `backend/grading_config.json`: grading thresholds and decision configuration.

# metrics.json schema

```json
{
  "model": "efficientnet_b0",
  "preprocessing": "clahe",
  "epochs": int,
  "trained_at": "ISO date",
  "accuracy": float,
  "macro_precision": float,
  "macro_recall": float,
  "macro_f1": float,
  "per_class": {
    "<class>": {
      "precision": f,
      "recall": f,
      "f1": f,
      "support": int
    }
  },
  "confusion_matrix": {
    "labels": [..],
    "matrix": [[..]]
  },
  "dataset": {
    "train": int,
    "val": int,
    "test": int,
    "per_class": {
      "<class>": {
        "train": int,
        "val": int,
        "test": int
      }
    }
  },
  "speed": {
    "cpu_ms_per_image": f,
    "cpu_images_per_sec": f,
    "gpu_ms_per_image": f_or_null,
    "gpu_images_per_sec": f_or_null
  },
  "ablation": {
    "baseline_accuracy": f,
    "clahe_accuracy": f
  } or null
}
```

# API contract

Built in Phase 4 and used by Phase 5:

- `GET /api/health` -> `{"status":"ok","model_loaded":bool,"classes":[..]}`
- `GET /api/metrics` -> the `reports/metrics.json` content.
- `GET /api/classes` -> `[{"name":str,"display_name":str,"description":str,"sample_images":["/static/samples/<class>/1.jpg", ...]}]`
- `POST /api/predict` (multipart field `file`) -> `{"filename":str,"predicted_class":str,"confidence":float,"probabilities":{"<class>":float},"grade":str,"decision":str,"inference_ms":float}`
- Errors: `400` for non-image files, `413` for files over 10 MB, JSON body `{"detail": "..."}`.

# Conventions

- Use Windows PowerShell commands and relative paths.
- Use seed `42`.
- Python code uses `pathlib`; do not hardcode absolute paths.
