from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path

from backend.endpoints import router

app = FastAPI(title="Delay Prediction API")
app.include_router(router)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).resolve().parents[1] / "frontend" / "index.html")
