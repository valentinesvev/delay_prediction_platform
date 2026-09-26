"""Последовательность для нейросети: форма и анти-утечка (будущее не влияет)."""

import numpy as np

from ml.feature_engineering import VehiclePlan, VehicleTrack
from ml.sequence import N_CHANNELS, SEQ_LEN, build_sequence

T0 = 1767700800.0


def _data(n_future=0):
    plan = VehiclePlan.from_arrays(np.arange(60), T0 - 1800 + 90 * np.arange(60),
                                   37.5 + 0.006 * np.arange(60), np.full(60, 55.75))
    t = T0 - 1800 + 15 * np.arange(121 + n_future)  # последняя «прошлая» точка ровно в T0
    track = VehicleTrack.from_arrays(t, 37.5 + 0.006 * (t - T0 + 1800) / 90, np.full(len(t), 55.75),
                                     np.full(len(t), 18.0), np.ones(len(t), bool))
    return plan, track


def test_shape():
    plan, track = _data()
    s = build_sequence(plan, track, T0, 0.0)
    assert s.shape == (SEQ_LEN, N_CHANNELS) and s.dtype == np.float32 and np.isfinite(s).all()


def test_future_telemetry_ignored():
    plan, track = _data()
    plan2, track2 = _data(n_future=40)
    assert np.array_equal(build_sequence(plan, track, T0, 30.0), build_sequence(plan2, track2, T0, 30.0))


def test_no_plan_no_track():
    empty = VehicleTrack.from_arrays([], [], [], [], [])
    s = build_sequence(None, empty, T0, 0.0)
    assert s.shape == (SEQ_LEN, N_CHANNELS)
