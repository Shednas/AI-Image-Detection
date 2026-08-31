# AI Image Detection, web application

FastAPI backend serving four detection models, React frontend for uploading
images and inspecting the result. Single images or a zip for batch processing,
with per-model visualisations and a searchable history.

## Requirements

- Python 3.14.3
- Node.js LTS
- PostgreSQL 18
- A CUDA GPU is optional. Everything runs on CPU, more slowly.

---

## Setup

Follow the steps in order, in a single terminal. Setup starts at the
repository root, but step 3 leaves you inside `app\backend`, so from there
on each step says where it expects you to be.

### Step 1: Create the database

Open pgAdmin, which is installed alongside PostgreSQL. Right-click **Databases**,
choose **Create**, and name it `ai_detection`.

If you prefer a terminal, use the full path to `psql`, since the PostgreSQL
installer does not add it to PATH on Windows:

```powershell
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -c "CREATE DATABASE ai_detection;"
```

Tables are created automatically on first startup.

### Step 2: Create the environment file

Copy `app\backend\.env.example` to `app\backend\.env` and set your PostgreSQL
password:

```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/ai_detection
```

The database name at the end must match the one created in step 1.

### Step 3: Install the backend

Open the VS code terminal:

```powershell
cd app\backend
python -m venv venv
venv\Scripts\activate
```

Install PyTorch first, from its own index.

With an NVIDIA GPU:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Without one:

```powershell
pip install torch torchvision
```

Then the rest:

```powershell
pip install -r ..\requirements.txt
```

Confirm it worked:

```powershell
python -c "import torch, lightgbm, skimage, scipy; print(torch.__version__, torch.cuda.is_available())"
```

### Step 4: Add the model weights

The four weight files are included in this package. Place them in
`app\backend\models\weights\`:

```
best_cnn.pt
best_fft.pt
best_hybrid.pt
stm_model.joblib
```

All four are Stage 3 checkpoints. Other stages produce numbers that disagree
with the dissertation.

### Step 5: Same terminal Install the frontend

```powershell
cd ..\frontend
npm install
```

If New terminal:
```powershell
cd app\frontend
npm install
```

npm reports several dependency advisories. These are in build tooling, not in
code that runs in the browser. Do not run `npm audit fix`.

---

## Run the application

Each server holds its terminal, so this needs two. Open two fresh ones
(in VS Code, **Terminal > New Terminal**). A new terminal opens at the
repository root, which is where both blocks below start. Do not reuse the
setup terminal, which is still down in `app\frontend`.

**New Terminal 1, backend:**

```powershell
cd app\backend
venv\Scripts\activate
python -m uvicorn main:app --reload --port 8000
```

**New Terminal 2, frontend:**

```powershell
cd app\frontend
npm run dev
```

Before using the application, open http://localhost:8000/api/health and confirm
it reports `"database": "up"` and all four models `true`.

A missing database does not stop the backend. It starts degraded: analysis still
returns a verdict, `/api/health` reports `"database": "down"` and answers 503,
and every affected result carries a `warning` field that the interface displays
above the result.

Then open **http://localhost:5173**.

First startup loads all four models, which takes a couple of minutes on CPU.

---

## About the application

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | One image, one model. Returns verdict, probability and visualisations. |
| `POST /api/batch` | Zip of images, one model. Returns per-file rows and a summary. |
| `GET /api/history` | Past results, with optional search and category filters. |
| `GET /api/health` | Per-model load state, database reachability, device. |

Limits: 10MB per image, 100 images per zip, 200MB total uncompressed.

Probabilities are reported as P(AI). Training used `{ai_generated: 0, real: 1}`,
so the raw sigmoid output is P(real). The conversion happens once, in
`pipeline.predict`.

---

## Additional information

**Why PyTorch installs separately.** An `--index-url` line inside
`requirements.txt` applies to every package in the file, not just torch.

**Why `python -m uvicorn` rather than `uvicorn`.** The `Scripts\*.exe` launchers
hard-code an absolute path to the interpreter, so they break if the folder is
moved. The module form does not.

**Known issues.** The interface does not poll `/api/health`, so it does not
show that the database is down until a result comes back carrying a warning.
Check the endpoint directly if History is not filling up.

Loading STM prints a scikit-learn `InconsistentVersionWarning`. `stm_model.joblib`
was fitted under scikit-learn 1.8.0, and `requirements.txt` pins no versions, so a
fresh install unpickles it under a newer one. It appears once, at startup, and
predictions are unaffected.

STM was fitted with named columns and is called with a positional array. The
feature order is identical, so predictions are unaffected. Do not change how the
features are passed: it risks altering predictions for a cosmetic gain.

**Running the tests.** From a fresh terminal at the repository root:

```powershell
cd app\backend
venv\Scripts\activate
pip install -r ..\requirements-dev.txt
pytest
```

Without a database, expect **45 passed, 3 skipped, 1 xfailed**. The three skips
are in `test_history_search.py`, which asserts PostgreSQL `ilike` escaping and
so skips when `DATABASE_URL` is unset rather than proving nothing against a
stub. With the database from step 1 reachable they run, giving **48 passed,
1 xfailed**. The xfail is expected either way. No test should fail.

---

## Troubleshooting

**`psql` is not recognised**
PostgreSQL is installed but not on PATH. Use pgAdmin, or the full path shown in
step 1.

**`DATABASE_URL is not set`**
`app\backend\.env` is missing. See step 2.

**`/api/health` returns 503**
Read the response body. It reports each model and the database separately. A
model reading `false` means its weights file is missing from
`app\backend\models\weights\`. `"database": "down"` means the database name or
password in `.env` does not match step 1.

**`port 8000 is already in use`**
Run `netstat -ano | findstr :8000`, then `taskkill /PID <pid>`. Or start uvicorn
on another port and update the CORS origin in `app\backend\main.py`.

**CORS errors in the browser**
The backend allows `http://localhost:5173` only. If Vite chose a different port,
update the origin in `app\backend\main.py`.