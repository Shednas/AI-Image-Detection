# Running the AI Image Detection Application

This guide explains how to start the database, backend server, and frontend development server.

---

## Prerequisites

Ensure you have completed the setup from `SETUP.md`:
- Python 3.14+ with virtual environment activated in `backend/`
- Node.js and npm installed
- PostgreSQL installed and running
- All dependencies installed (`pip install -r requirements.txt`, `npm install`)
- `.env` file configured in `backend/` with your PostgreSQL password

---

## Step 1: Start PostgreSQL Database

PostgreSQL should run as a Windows service automatically. Verify it's running:

**Option A: Check Services (Recommended)**
1. Press `Win + R`, type `services.msc`, and press Enter
2. Look for `postgresql-x64-15` (or your version)
3. If it's not running, right-click and choose **Start**

**Option B: Start via Command Line**
```powershell
"C:\Program Files\PostgreSQL\15\bin\pg_ctl" -D "C:\Program Files\PostgreSQL\15\data" start
```
(Replace `15` with your PostgreSQL version if different)

**Verify the database exists:**
```powershell
"C:\Program Files\PostgreSQL\15\bin\psql" -U postgres -c "SELECT datname FROM pg_database WHERE datname='ai_detection';"
```

If `ai_detection` doesn't exist, create it:
```powershell
"C:\Program Files\PostgreSQL\15\bin\psql" -U postgres
```
Then run:
```sql
CREATE DATABASE ai_detection;
\q
```

---

## Step 2: Start Backend Server

Open a PowerShell terminal and navigate to the backend:

```powershell
cd backend
venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
Loading models on cuda...
  CNN loaded
  FFT loaded
  Hybrid loaded
  STM loaded
All models ready.
INFO:     Application startup complete.
```

**Keep this terminal open.** The backend will automatically reload when you make code changes.

Test the backend health:
```
http://localhost:8000/api/health
```

---

## Step 3: Start Frontend Server

Open a **second** PowerShell terminal and navigate to the frontend:

```powershell
cd frontend
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  press h + enter to show help
```

**Keep this terminal open.** The frontend will automatically reload when you make code changes.

---

## Step 4: Open the Application

1. Open your browser and go to: **http://localhost:5173**
2. You should see the AI Image Detection interface
3. Upload an image to analyze
4. Select a model (CNN, FFT, Hybrid, or STM)
5. Click "Analyze" to run detection

---

## Running Both Simultaneously

You need **two terminal windows open at the same time:**

| Terminal 1 | Terminal 2 |
|---|---|
| Backend (port 8000) | Frontend (port 5173) |
| `cd backend` | `cd frontend` |
| `venv\Scripts\activate` | (no venv needed) |
| `uvicorn main:app --reload --port 8000` | `npm run dev` |

Both will automatically reload on file changes.

---

## Troubleshooting

**"Could not connect to server" (PostgreSQL)**
- Ensure PostgreSQL service is running (see Step 1)
- Check that the password in `.env` matches your PostgreSQL password

**"Address already in use: port 8000"**
- Another process is using port 8000. Either:
  - Kill the process: `netstat -ano | findstr :8000` then `taskkill /PID <PID>`
  - Change the port: `uvicorn main:app --reload --port 8001`

**"Address already in use: port 5173"**
- Kill the process using port 5173 or let Vite find an available port automatically

**Models taking a long time to load**
- First startup loads all model weights (~2-5 minutes). This is normal, especially on slower GPUs.
- Wait for "All models ready." message before accessing the API

**CORS error in browser console**
- Make sure backend is on port 8000 and frontend on port 5173
- If you changed either port, update the CORS origin in `backend/main.py` line 25

**"No such file or directory: best_cnn.pt"**
- Model weights should be in `backend/models/weights/`
- Check that all four files exist: `best_cnn.pt`, `best_fft.pt`, `best_hybrid.pt`, `stm_model.joblib`

---

## Stopping Everything

Press `CTRL+C` in each terminal to stop:
1. Backend server (Terminal 1)
2. Frontend server (Terminal 2)
3. PostgreSQL (optional, it runs as a service)

---

## Environment Variables

Your `.env` file should look like:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/ai_detection
```

Replace `YOUR_PASSWORD` with your actual PostgreSQL password set during installation.

---

## Quick Start Script (Optional)

Create a file called `run.bat` in the project root to start both with one click:

```batch
@echo off
start cmd /k "cd backend && venv\Scripts\activate && uvicorn main:app --reload --port 8000"
start cmd /k "cd frontend && npm run dev"
echo Backend running on http://localhost:8000
echo Frontend running on http://localhost:5173
pause
```

Then double-click `run.bat` to start both servers.
