# AI Image Detection, web application

FastAPI backend serving four detection models, React frontend for uploading
images and inspecting the result. Single images or a zip for batch processing,
with per-model visualisations and a searchable history.

## Requirements

- Python 3.14.3
- Node.js LTS
- PostgreSQL 18
- A CUDA GPU is optional. Everything runs on CPU, more slowly.

## Setup

### 1. Database

Create the database. Any PostgreSQL client will do:

```powershell
psql -U postgres -c "CREATE DATABASE ai_detection;"
```

Tables are created on first startup, so there is no migration step to run.

### 2. Backend

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r ..\requirements.txt
```

Copy `backend/.env.example` to `backend/.env` and set your password:

```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/ai_detection
```

The path is resolved relative to `backend/`, so it is found regardless of which
directory you start uvicorn from.

### 3. Model weights

Download the four files from the release and put them in
`backend/models/weights/`:

```
best_cnn.pt
best_fft.pt
best_hybrid.pt
stm_model.joblib
```

All four are the Stage 3 checkpoints. Using a different stage will produce
numbers that disagree with the dissertation.

### 4. Frontend

```powershell
cd frontend
npm install
```

## Running

Two terminals.

Backend:

```powershell
cd backend
venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

Frontend:

```powershell
cd frontend
npm run dev
```

Then open http://localhost:5173.

First startup loads all four models, which takes a couple of minutes on CPU.
`GET /api/health` returns 200 once every model is loaded and the database
answers, and 503 with a breakdown of what is missing otherwise. Check it before
assuming something is broken.

## Tests

```powershell
cd backend
venv\Scripts\activate
pip install -r ..\requirements.txt -r ..\requirements-dev.txt
pytest
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | One image, one model. Returns verdict, probability and visualisations. |
| `POST /api/batch` | Zip of images, one model. Returns per-file rows and a summary. |
| `GET /api/history` | Past results, with optional search and category filters. |
| `GET /api/health` | Per-model load state, database reachability, device. |

Probabilities are reported as P(AI). Training used `{ai_generated: 0, real: 1}`,
so the raw sigmoid is P(real); the conversion happens once, in `pipeline.predict`.

Limits: 10MB per image, 100 images per zip, 200MB total uncompressed.

## Troubleshooting

**`DATABASE_URL is not set`**
`backend/.env` is missing. Copy `.env.example` and fill in the password.

**Backend starts but `/api/health` returns 503**
Read the body. It reports each model separately and the database state. A model
reading `false` means its weights file is missing or unreadable.

**`port 8000 is already in use`**
`netstat -ano | findstr :8000`, then `taskkill /PID <pid>`, or start uvicorn on
another port and update the CORS origin in `backend/main.py`.

**CORS errors in the browser**
The backend allows `http://localhost:5173` only. If Vite picked a different
port, update the origin.
