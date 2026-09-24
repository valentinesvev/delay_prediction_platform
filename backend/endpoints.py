from fastapi import APIRouter
from pydantic import BaseModel, FiniteFloat

from ml.features import build_features
from ml.predictor import BaselinePredictor

router = APIRouter()
predictor = BaselinePredictor()


class PredictionRequest(BaseModel):
    current_delay: FiniteFloat


class PredictionResponse(BaseModel):
    predicted_delay: FiniteFloat


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    features = build_features(request.current_delay)
    return PredictionResponse(predicted_delay=predictor.predict(features))
