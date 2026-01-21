## Frontend quickstart

This frontend calls the FastAPI backend and displays the generated image.

1. Open Terminal 1 and start the backend.

```bash
cd /home/hhdel/repos/MLOps2026/mlops2026
source .venv/bin/activate
PYTHONPATH=src uvicorn vdm_pokemon.api:app --reload
```

2. Open Terminal 2 and start the frontend.

```bash
cd /home/hhdel/repos/MLOps2026/mlops2026
source .venv/bin/activate
BACKEND=http://127.0.0.1:8000 streamlit run frontend.py
```

3. Open http://localhost:8501 in a browser, press Generate, and watch the backend terminal for POST /generate 200 OK.
