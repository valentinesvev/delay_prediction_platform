from typing import Protocol

from ml.features import DelayFeatures


class Predictor(Protocol):
    """Единый интерфейс baseline и будущей ML-модели."""

    def predict(self, features: DelayFeatures) -> float:
        ...


class BaselinePredictor:
    def predict(self, features: DelayFeatures) -> float:
        """Прогноз будущей задержки равен текущей задержке в минутах."""
        return features.current_delay
