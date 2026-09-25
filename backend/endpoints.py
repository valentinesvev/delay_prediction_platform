import os
import time
from datetime import datetime, timezone
from fastapi import APIRouter
from data.storage import latest_prediction, average_latest_predictions

router = APIRouter()


@router.get('/health')
def health():
    return {'status': 'ok'}


@router.get('/predictions/latest')
def get_latest_prediction():
    record = latest_prediction()
    if record is None:
        return {'status': 'unavailable', 'state': None}
    max_age = float(os.getenv('PREDICTION_MAX_AGE_SECONDS', '15'))
    stale = time.time() - min(record['predicted_at'], record['data_as_of']) > max_age
    return {
        'status': 'stale' if stale else 'ready',
        'state': record['state'],
        'delay': record['delay'],
        'predicted_at': datetime.fromtimestamp(record['predicted_at'], timezone.utc).isoformat(),
        'data_as_of': datetime.fromtimestamp(record['data_as_of'], timezone.utc).isoformat(),
    }


@router.get('/predictions/average')
def get_average_prediction():
    """Среднее state по последним 10 сохранённым прогнозам (или всем, если их меньше)."""
    result = average_latest_predictions()
    return {'status': 'ready' if result['count'] else 'unavailable', **result}
