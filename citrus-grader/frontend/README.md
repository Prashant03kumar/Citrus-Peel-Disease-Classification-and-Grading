# Citrus Disease Grader dashboard

The React dashboard reads metrics, class samples, health, and predictions from the FastAPI backend. Configure the API URL in `.env` (see `.env.example`).

Run both processes from the project root in separate terminals:

```powershell
.\backend\run_backend.ps1
Set-Location frontend
npm run dev
```

The dashboard is available at `http://localhost:5173`. Build and lint it from `frontend/`:

```powershell
npm run build
npm run lint
```
