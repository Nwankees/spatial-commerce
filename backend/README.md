# Visual analysis backend

This local FastAPI service keeps the Gemini credential off the Android device and returns a validated `VisualProductAnalysis` JSON object.

## Local setup

From `backend/`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# Put the HackGT Gemini key in .env (never commit this file).
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health`. For the USB-connected Android device, run:

```powershell
adb reverse tcp:8000 tcp:8000
```

The debug Android app then reaches the service at `http://127.0.0.1:8000` through the USB connection.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

The API documentation is available at `http://127.0.0.1:8000/docs` while the service is running.
