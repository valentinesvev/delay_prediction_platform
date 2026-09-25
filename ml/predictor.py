"""Выбор модели и запасного прогноза; без операций с БД."""
import logging
import os


logger = logging.getLogger(__name__)
MODEL_TYPE = os.getenv('MODEL_TYPE', 'baseline').strip().lower()


class ModelUnavailableError(RuntimeError):
    """Выбранная модель пока не подключена."""


def predict_catboost(features):
    raise ModelUnavailableError('CatBoost пока не подключён')


def predict_torch(features):
    raise ModelUnavailableError('Torch пока не подключён')


def predict_baseline(features):
    """Учебный прогноз: исходные состояние и задержка."""
    return {'state': features['state'], 'delay': features['delay']}


def predict(features):
    """Выбрать модель по MODEL_TYPE; при её ошибке вернуть baseline."""
    try:
        if MODEL_TYPE == 'catboost':
            result = predict_catboost(features.copy())
            logger.info('prediction source=catboost')
            return result
        if MODEL_TYPE == 'torch':
            result = predict_torch(features.copy())
            logger.info('prediction source=torch')
            return result
        if MODEL_TYPE != 'baseline':
            logger.warning('Неизвестный MODEL_TYPE=%r; используем baseline', MODEL_TYPE)
    except Exception as exc:
        logger.warning('Ошибка модели %s; используем baseline: %s', MODEL_TYPE, exc,
                       exc_info=not isinstance(exc, ModelUnavailableError))
    result = predict_baseline(features)
    logger.info('prediction source=baseline')
    return result
