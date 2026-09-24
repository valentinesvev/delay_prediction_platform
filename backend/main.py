from fastapi import FastAPI

from backend.endpoints import router

app = FastAPI(title="Delay Prediction API")
app.include_router(router)
