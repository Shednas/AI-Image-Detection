# Setup Guide: AI Image Detection (Windows)

---

## Step 1: Install Node.js

Download the LTS version from https://nodejs.org and run the installer.

Verify:
```
node --version
npm --version
```

---

## Step 2: Install PostgreSQL

Download from https://www.postgresql.org/download/windows and run the installer.

During installation:
- Set a password for the `postgres` user (remember this)
- Keep default port: 5432
- Include pgAdmin 4 in the installation (useful GUI)

Create the database after installation:

**Option A, pgAdmin 4:**
Open pgAdmin, right-click "Databases", then Create, then Database, then name it `ai_detection`

**Option B, command line:**
```
psql -U postgres
CREATE DATABASE ai_detection;
\q
```

---

## Step 3: Backend Setup

Open a terminal and navigate to your backend folder.

### 3a. Create and activate virtual environment
```
cd backend
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

### 3b. Install dependencies
```
pip install fastapi "uvicorn[standard]" sqlalchemy psycopg2-binary python-multipart python-dotenv pillow numpy
```

Note: torch, torchvision, lightgbm, scikit-image, and scipy should already be installed from your research work. If not:
```
pip install torch torchvision lightgbm scikit-image scipy
```

### 3c. Create .env file
Copy `.env.example` to `.env` and fill in your PostgreSQL password:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/ai_detection
```

### 3d. Fix hybrid_model.py import
Open `backend/models/hybrid_model.py` and change the import on line 3 from:
```python
from src.models.fft_model import FFTDetector
```
to:
```python
from models.fft_model import FFTDetector
```
This is required because the web app runs from the `backend/` directory, not the research `src/` structure.

### 3e. Place model weights
Make sure these files exist in `backend/models/weights/`:
```
best_cnn.pt
best_fft.pt
best_hybrid.pt
stm_model.joblib     (your trained STM checkpoint from train_stm.py)
```

### 3f. Run the backend
```
uvicorn main:app --reload --port 8000
```

The backend runs at http://localhost:8000
The database tables are created automatically on first startup.
Check http://localhost:8000/api/health, which should return `{"status":"ok"}`.

---

## Step 4: Frontend Setup

Open a second terminal and navigate to your frontend folder.

### 4a. Create Vite + React project
```
cd frontend
npm create vite@latest . -- --template react
```
If prompted about existing files, confirm to proceed.

### 4b. Install dependencies
```
npm install
npm install react-router-dom axios
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### 4c. Copy the provided files
Replace or create the following files:

| File | Action |
|---|---|
| src/App.jsx | Replace the generated file |
| src/index.css | Replace (just Tailwind directives) |
| tailwind.config.js | Replace (update content array) |
| src/pages/AnalyzePage.jsx | Create new folder + file |
| src/pages/BatchPage.jsx | New file |
| src/pages/HistoryPage.jsx | New file |
| src/pages/FaqPage.jsx | New file |
| src/api/api.js | Create new folder + file |

### 4d. Run the frontend
```
npm run dev
```

The frontend runs at http://localhost:5173

---

## Running Both Together

You need two terminals open simultaneously:

**Terminal 1 (backend):**
```
cd backend
venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 (frontend):**
```
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

---

## Troubleshooting

**"No module named 'database'"**
Make sure you are running uvicorn from inside the `backend/` folder, not from the project root.

**"DATABASE_URL is not set"**
Copy `backend/.env.example` to `backend/.env` and put your PostgreSQL password in it. The file is read from `backend/` regardless of which directory you start uvicorn in.

**"could not connect to server" (PostgreSQL)**
Ensure the PostgreSQL service is running. Open Windows Services and check that "postgresql-x64-XX" is running.

**"No .joblib file found in weights directory"**
Check that your STM checkpoint file ends in `.joblib` and is in `backend/models/weights/`.

**CORS error in browser**
Make sure the backend is running on port 8000 and the frontend on port 5173. If you change either port, update the CORS origin in `backend/main.py`.

**Models take a long time to load on startup**
This is normal on first run, especially on CPU. The startup message "All models ready." confirms all four loaded successfully.
