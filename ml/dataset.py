"""Загрузка CSV хакатона и сборка таблицы признаков для train / test / validate.

Использование::

    python -m ml.dataset --data ../dataset --out artifacts/features

Из ``schedule.csv`` берутся **только плановые** колонки (``time_begin``, ``geom``,
``manual_fill``) — так признаки train/test/validate считаются одинаково и не
зависят от факта прибытия.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.feature_engineering import FEATURE_NAMES, VehiclePlan, VehicleTrack, compute_features

SPLITS = {
    # split: (traffic, schedule, points)
    "train": ("train/traffic.csv", "train/schedule.csv", "labels/labels_train.csv"),
    "test": ("test/traffic.csv", "test/schedule.csv", "labels/labels_test.csv"),
    "validate": ("validate/traffic.csv", "validate/schedule_plan.csv", "validate/points.csv"),
}


def _to_unix(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).astype("datetime64[ns]").astype("int64").to_numpy() / 1e9


def load_plans(path: Path) -> dict[int, VehiclePlan]:
    """Прочитать расписание (только план) и разбить по ТС."""
    sc = pd.read_csv(path, usecols=["tt_action_item_id", "tr_id", "time_begin", "geom", "manual_fill"])
    xy = sc["geom"].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    sc["lon"], sc["lat"] = xy[0], xy[1]
    sc["t"] = _to_unix(sc["time_begin"])
    plans = {}
    for tr_id, g in sc.groupby("tr_id"):
        plans[int(tr_id)] = VehiclePlan.from_arrays(
            g["tt_action_item_id"].to_numpy(), g["t"].to_numpy(),
            g["lon"].to_numpy(), g["lat"].to_numpy(), g["manual_fill"].astype(float).to_numpy(),
        )
    return plans


def load_tracks(path: Path) -> dict[int, VehicleTrack]:
    """Прочитать телеметрию и разбить по ТС.

    Для анти-утечки время точки = ``max(event_time, receive_time)``: точка
    считается доступной только когда она и произошла, и была получена.
    """
    cols = ["tr_id", "event_time", "receive_time", "location_valid", "lon", "lat", "speed"]
    tr = pd.read_csv(path, usecols=cols, low_memory=False)
    t_ev = _to_unix(tr["event_time"])
    t_rx = _to_unix(tr["receive_time"])
    tr["t"] = np.fmax(t_ev, t_rx)
    tr["valid"] = tr["location_valid"].astype(str).str.lower().eq("true")
    tracks = {}
    for tr_id, g in tr.groupby("tr_id"):
        tracks[int(tr_id)] = VehicleTrack.from_arrays(
            g["t"].to_numpy(), g["lon"].to_numpy(), g["lat"].to_numpy(),
            g["speed"].to_numpy(), g["valid"].to_numpy(),
        )
    return tracks


def build_split(data_dir: Path, split: str, use_cur_dev: bool = True) -> pd.DataFrame:
    """Собрать таблицу признаков для одного сплита.

    Args:
        data_dir: папка ``dataset`` хакатона.
        split: ``train`` | ``test`` | ``validate``.
        use_cur_dev: передавать ли подсказку ``cur_dev_s`` (False — режим
            «чистого потока», когда подсказки нет).
    """
    traffic, schedule, points = SPLITS[split]
    plans = load_plans(data_dir / schedule)
    tracks = load_tracks(data_dir / traffic)
    pts = pd.read_csv(data_dir / points)
    pts["T_unix"] = _to_unix(pts["T"])
    empty = VehicleTrack.from_arrays([], [], [], [], [])

    rows = []
    for r in pts.itertuples(index=False):
        plan = plans[int(r.tr_id)]
        hit = np.where(plan.stop_ids == r.target_stop_id)[0]
        tidx = int(hit[0]) if len(hit) else None
        f = compute_features(
            plan, tracks.get(int(r.tr_id), empty), float(r.T_unix), tidx,
            float(r.cur_dev_s) if use_cur_dev else None,
        )
        f["sample_id"] = r.sample_id
        f["tr_id"] = int(r.tr_id)
        f["T"] = r.T
        f["cur_dev_given"] = float(r.cur_dev_s)
        if hasattr(r, "target_delay_s"):
            f["target_delay_s"] = float(r.target_delay_s)
        rows.append(f)
    df = pd.DataFrame(rows)
    df["split"] = split
    df["synthetic"] = (df["tr_id"] >= 9_000_000).astype(int)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/features"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for mode, use in [("hint", True), ("stream", False)]:
            df = build_split(args.data, split, use)
            df.to_csv(args.out / f"{split}_{mode}.csv", index=False)
            print(split, mode, df.shape, "features:", len(FEATURE_NAMES))


if __name__ == "__main__":
    main()
