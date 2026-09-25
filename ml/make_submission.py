"""Сформировать ``submission.csv`` для раздела «Data Science».

Использование::

    python -m ml.make_submission --data ../dataset --out submission.csv

Берёт порядок ``sample_id`` из ``sample_submission.csv``, признаки validate
(режим ``hint`` — с подсказкой ``cur_dev_s``) и ансамбль из ``artifacts/model``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.dataset import build_split
from ml.predictor import DEFAULT_MODEL_DIR, DelayPredictor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="папка dataset хакатона")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = ap.parse_args()

    feats = build_split(args.data, "validate", use_cur_dev=True)
    predictor = DelayPredictor(args.model)
    seqs = None
    if predictor.uses_nn:
        from ml.dataset import SPLITS, load_plans, load_tracks
        from ml.train_nn import sequences_for
        traffic, schedule, _ = SPLITS["validate"]
        seqs = sequences_for(load_plans(args.data / schedule), load_tracks(args.data / traffic), feats)
    pred = predictor.predict_frame(feats, "hint", seqs)["delay"]
    print("NN в ансамбле:", predictor.uses_nn)
    by_id = dict(zip(feats["sample_id"], np.round(pred, 1)))

    sub = pd.read_csv(args.data / "sample_submission.csv", sep=";")
    missing = set(sub["sample_id"]) - set(by_id)
    if missing:
        raise SystemExit(f"Нет прогноза для {len(missing)} sample_id")
    sub["prediction"] = sub["sample_id"].map(by_id)
    assert sub["prediction"].notna().all() and not sub["sample_id"].duplicated().any()
    sub.to_csv(args.out, sep=";", index=False)
    print(f"saved {args.out}: {len(sub)} rows; mean={sub.prediction.mean():.1f}, "
          f"baseline mean={feats['cur_dev_given'].mean():.1f}")


if __name__ == "__main__":
    main()
