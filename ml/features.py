from dataclasses import dataclass


@dataclass(frozen=True)
class DelayFeatures:
    """Признаки прогноза; задержка измеряется в минутах."""

    current_delay: float


def build_features(current_delay: float) -> DelayFeatures:
    """Начальный pipeline: только текущая задержка, без преобразований."""
    return DelayFeatures(current_delay=current_delay)
