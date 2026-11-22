# Project progress

| Phase | Scope | Status | Date | Notes |
|---|---|---|---|---|
| 1 | Environment and project skeleton | DONE | 2026-11-22 | Created Python environment, shared requirements, Vite React app, and project contracts. |
| 2 | Dataset preparation and statistics | DONE | 2026-11-22 | Classes: black-spot and citrus-canker. Counts (train/val/test): black-spot 875/155/206; citrus-canker 851/151/201. No corrupt images, non-image files, or train/test duplicates. |
| 3 | Model training and evaluation | DONE | 2026-11-22 | Full run on CPU with early stopping: baseline test accuracy 98.53%; CLAHE test accuracy 98.53%, macro F1 98.53%. Per-class F1: black-spot 98.54%, citrus-canker 98.51%. CPU inference 91.92 ms/image (10.88 images/sec); GPU unavailable. Test split was used for evaluation only, never training or model selection. |
| 4 | FastAPI prediction backend | NOT STARTED | — | Load model artifacts, expose API contracts, and configure grading and sample images. |
| 5 | React dashboard | NOT STARTED | — | Build metrics, dataset, class samples, and photo-upload prediction views. |
| 6 | Integration and validation | NOT STARTED | — | Verify the complete application workflow and document final usage. |
