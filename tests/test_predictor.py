import pytest

from ml.features import build_features
from ml.predictor import BaselinePredictor, Predictor


@pytest.mark.parametrize("current_delay", [0.0, 12.5, -3.0])
def test_baseline_preserves_current_delay(current_delay):
    predictor: Predictor = BaselinePredictor()
    assert predictor.predict(build_features(current_delay)) == current_delay
