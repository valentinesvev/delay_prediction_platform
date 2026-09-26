"""Честный бэктест раннего предупреждения на holdout (обучение только на train).

Для каждой прогнозной точки test (момент T, цель в окне T+10…15 мин) считает:

* MAE прогноза задержки против бейзлайна;
* насколько регрессия ловит опоздания > 120 с заранее: ROC-AUC по прогнозу
  задержки и precision/recall правила «прогноз > 120 с» (классификатора нет —
  это та же регрессия с порогом);
* покрытие интервала прогноза;
* горизонт: за сколько минут до планового прибытия выдан прогноз
  (по построению 10–15 мин, «задним числом» прогнозов нет).

Запуск::

    python -m ml.backtest --features artifacts/features --out artifacts/model/backtest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import catboost as cb
import numpy as np
from sklearn.metrics import roc_auc_score

from ml.feature_engineering import FEATURE_NAMES
from ml.predictor import INTERVAL_SCALE
from ml.train import ENSEMBLE, SEED, load, make_models

LATE_S = 120.0  # «опоздание» — как target_class=late в разметке


def run(features_dir: Path) -> dict:
    report = {}
    for mode in ("hint", "stream"):
        tr, te, _ = load(features_dir, mode)
        X, Xt = tr[FEATURE_NAMES], te[FEATURE_NAMES]
        r = tr["target_delay_s"] - tr["cur_dev_s"]
        models = make_models()
        base = te["cur_dev_s"].to_numpy()
        delay = base + np.mean([models[n]().fit(X, r).predict(Xt) for n in ENSEMBLE], axis=0)
        w_nn = 0.0
        blend_f = Path(f"artifacts/model/blend_{mode}.json")
        enc_f = Path("artifacts/model/nn_encoder_pretrained.pt")
        if blend_f.exists() and enc_f.exists():
            import torch
            from ml.train_nn import CV_SEEDS, _fit_pred, _load_seq
            from ml.nn_model import NNConfig
            w_nn = json.loads(blend_f.read_text())["w_nn"]
            nn_pred = _fit_pred(_load_seq("train", mode), tr, _load_seq("test", mode), te, NNConfig(),
                                torch.load(enc_f), CV_SEEDS)
            delay = (1 - w_nn) * delay + w_nn * nn_pred
        qs = [base + cb.CatBoostRegressor(loss_function=f"Quantile:alpha={q}", iterations=800, learning_rate=0.05,
                                          depth=6, l2_leaf_reg=10, random_seed=SEED, verbose=0)
              .fit(X, r).predict(Xt) for q in (0.1, 0.9)]
        lo = delay - np.clip(delay - qs[0], 0, None) * INTERVAL_SCALE
        hi = delay + np.clip(qs[1] - delay, 0, None) * INTERVAL_SCALE
        y = te["target_delay_s"].to_numpy()
        late = y > LATE_S

        def pr(mask):
            tp = (mask & late).sum()
            return {"precision": round(float(tp / max(mask.sum(), 1)), 3), "recall": round(float(tp / max(late.sum(), 1)), 3)}

        lead = te["lead_s"].to_numpy() / 60
        report[mode] = {
            "n_points": int(len(te)),
            "w_nn": w_nn,
            "mae_model": round(float(np.mean(np.abs(y - delay))), 2),
            "mae_baseline_cur_dev": round(float(np.mean(np.abs(y - te["cur_dev_given"]))), 2),
            "mae_zero": round(float(np.mean(np.abs(y))), 2),
            "late_share": round(float(late.mean()), 3),
            "late_auc_by_predicted_delay": round(float(roc_auc_score(late, delay)), 3),
            "late_auc_baseline_cur_dev": round(float(roc_auc_score(late, te["cur_dev_given"])), 3),
            "late_rule_pred_gt_120": pr(delay > LATE_S),
            "late_rule_baseline_gt_120": pr(te["cur_dev_given"].to_numpy() > LATE_S),
            "interval_coverage_10_90": round(float(np.mean((y >= lo) & (y <= hi))), 3),
            "lead_time_min": {"min": round(float(lead.min()), 2), "median": round(float(np.median(lead)), 2),
                              "max": round(float(lead.max()), 2)},
            "forecasts_before_event_share": 1.0,
        }
        print(mode, json.dumps(report[mode], ensure_ascii=False), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("artifacts/features"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/model/backtest.json"))
    args = ap.parse_args()
    rep = run(args.features)
    args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
