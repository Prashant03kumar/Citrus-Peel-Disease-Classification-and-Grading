# Citrus Grader

Citrus Grader classifies visible citrus-peel conditions from a photo and returns a sorting recommendation. The project includes a trained EfficientNet-B0 model, a FastAPI inference API, and a React dashboard.

## Features

- EfficientNet-B0 image classification with CLAHE preprocessing.
- Model and class metadata loaded by the backend at startup.
- Per-class probabilities, confidence, and inference timing.
- Configurable grade/decision rules and a low-confidence manual-review threshold.
- Dashboard for model metrics, dataset splits, class examples, and photo predictions.
- Upload checks for image type and the 10 MB API limit.

## Architecture

```text
Browser
  React + Vite dashboard
       │ HTTP: metrics, classes, health, multipart prediction
       ▼
  FastAPI backend
       │ CLAHE → resize/normalize → inference
       ▼
  PyTorch EfficientNet-B0 model
       │
       └── class probabilities → configured grade/decision → dashboard
```

The API contract is recorded in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md). Interactive API documentation is available at `http://localhost:8000/docs`.

## Project structure

```text
citrus-grader/
├── backend/
│   ├── app/                 # FastAPI routes, inference, grading, schemas
│   ├── static/samples/      # Tracked class-gallery photos
│   ├── tests/               # API tests
│   ├── grading_config.json  # Editable confidence and grading rules
│   └── requirements.txt
├── data/                    # Local-only; not included in Git
├── frontend/                # React/Vite dashboard
├── ml/                      # Dataset preparation, training, evaluation
├── models/
│   ├── citrus_effb0.pth     # Tracked final inference weights
│   └── classes.json         # Tracked output class order
├── reports/                 # Tracked metrics, dataset stats, logs
├── check_setup.py
├── start_all.ps1
├── start_backend.ps1
└── start_frontend.ps1
```

`data/` is deliberately not in the repository because the image dataset is local and can be large. To prepare or retrain, put class folders under `data/train/<class>/` and `data/test/<class>/`. Phase 2 creates `data/val/<class>/` when needed.

## Prerequisites

- Windows PowerShell.
- Python 3.10 or newer (Python 3.11 was used for this project).
- Node.js 20.19+ or 22.12+ and npm (required by the Vite version in this project).
- Enough disk space for PyTorch and the Python dependencies.
- For retraining, the citrus image dataset arranged by class as described above.

## One-time setup (Windows)

Run these commands from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

Set-Location frontend
npm install
Copy-Item .env.example .env
Set-Location ..

python check_setup.py
```

If PowerShell prevents activation, open a PowerShell session that permits local scripts or run the venv interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe check_setup.py
```

`frontend/.env` sets `VITE_API_URL=http://localhost:8000`. It is local-only and ignored by Git; `.env.example` is tracked.

## Prepare data and train

The raw dataset is not downloaded or committed by this project. Either copy it into the project:

```text
data/
├── train/
│   ├── black-spot/
│   └── citrus-canker/
└── test/
    ├── black-spot/
    └── citrus-canker/
```

or point the preparation script at an existing dataset folder containing `train/` and `test/`:

```powershell
python -m ml.prepare_data --source-dir "..\your-citrus-dataset"
```

When the class split folders are already in `data/`, use:

```powershell
python -m ml.prepare_data
```

The preparation step validates images, creates a seeded validation split if absent, records counts, and writes class sample photos. Do not mix classes within a folder; folder names define the class labels.

Run the complete baseline/CLAHE training, evaluation, benchmark, and metrics pipeline with:

```powershell
python -m ml.run_all
```

This full run can take substantial time on CPU. For a development smoke run only, use `python -m ml.run_all --quick`; the quick run trains on a subset and overwrites generated model/report outputs. To train only one CLAHE model:

```powershell
python -m ml.train --clahe
```

The full pipeline writes the selected model to `models/citrus_effb0.pth`, metrics to `reports/metrics.json`, and the class order to `models/classes.json`. Keep the test split out of training and model selection; it is used only for the final reported evaluation.

## Run the application

From the project root, launch both services in separate PowerShell windows:

```powershell
.\start_all.ps1
```

Open the dashboard at <http://localhost:5173> and the Swagger API docs at <http://localhost:8000/docs>. To start the services individually:

```powershell
.\start_backend.ps1
```

```powershell
.\start_frontend.ps1
```

The backend runs at port 8000 and the frontend at port 5173. To run frontend checks from its folder:

```powershell
Set-Location frontend
npm run build
npm run lint
```

Run backend tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -q
```

## API reference

All API routes use the `/api` prefix. Responses follow the contract in `PROJECT_CONTEXT.md`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Readiness, loaded-model status, and class names |
| `GET` | `/api/metrics` | Contents of `reports/metrics.json` |
| `GET` | `/api/classes` | Class display names, descriptions, and sample-image URLs |
| `POST` | `/api/predict` | Classify a multipart upload with field name `file` |

Example health response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["black-spot", "citrus-canker"]
}
```

Example prediction request in PowerShell:

```powershell
curl.exe -F "file=@data\test\black-spot\black-spot  (1).jpg;type=image/jpeg" http://localhost:8000/api/predict
```

Example prediction response (timing and confidence vary by image and machine):

```json
{
  "filename": "black-spot  (1).jpg",
  "predicted_class": "black-spot",
  "confidence": 0.99999,
  "probabilities": {
    "black-spot": 0.99999,
    "citrus-canker": 0.00001
  },
  "grade": "B",
  "decision": "GRADE B (processing / juice)",
  "inference_ms": 90.0
}
```

Invalid/non-image uploads return HTTP 400, uploads larger than 10 MB return HTTP 413, and unavailable metrics return HTTP 503. Error responses use `{"detail":"..."}`.

## Edit grading rules

Edit `backend/grading_config.json`; rules are read for each prediction, so changes take effect without changing Python code:

```json
{
  "confidence_threshold": 0.90,
  "rules": {
    "citrus-canker": {
      "grade": "Reject",
      "decision": "REJECT (quarantine disease)"
    },
    "black-spot": {
      "grade": "B",
      "decision": "GRADE B (processing / juice)"
    },
    "healthy": {
      "grade": "A",
      "decision": "GRADE A (fresh market)"
    },
    "scab": {
      "grade": "B",
      "decision": "GRADE B (processing / juice)"
    }
  }
}
```

Predictions below the confidence threshold and unconfigured class names receive `Review` / `MANUAL INSPECTION`. The current model knows only black-spot and citrus-canker, so it can return Grade B, Reject, or Review; Grade A requires a future healthy class and a newly trained model.

## Model results

Values below are copied from `reports/metrics.json`. Precision, recall, and F1 are shown as percentages; support is the number of test images for that class.

| Class / comparison | Accuracy | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| Black Spot | — | 99.02% | 98.06% | 98.54% | 206 |
| Citrus Canker | — | 98.03% | 99.00% | 98.51% | 201 |
| Macro average (CLAHE) | 98.53% | 98.52% | 98.53% | 98.53% | 407 |
| Baseline (without CLAHE) | 98.53% | — | — | — | — |
| CLAHE (with CLAHE) | 98.53% | — | — | — | — |

CPU inference is **91.92 ms/image** (**10.88 images/second**). GPU timing is unavailable in the recorded run. Baseline and CLAHE test accuracy are equal in the current ablation.

## Known limitations

- The dataset currently contains 1,726 training, 306 validation, and 407 test images across two classes; performance may not generalize to different orchards, cameras, lighting, or fruit varieties.
- Only `black-spot` and `citrus-canker` are supported by the trained model. The healthy and scab grading rules do not add those classes to the model.
- The API accepts one fruit image per prediction; images with multiple fruits may be ambiguous.
- This is a sorting aid, not a certified plant-health diagnosis, food-safety assessment, or substitute for expert inspection. Low-confidence results require manual review.

## Future work

- Add Grad-CAM explanations for model predictions.
- Collect and label healthy and scab examples, then retrain and evaluate a model with those classes.
- Export the model to ONNX for additional runtimes and deployment options.
- Integrate a camera or conveyor-belt workflow for automated sorting.
