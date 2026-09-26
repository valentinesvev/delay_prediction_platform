"""Сравнение моделей, выбор лучшей и обучение финальных артефактов.

Схема валидации (честная, без утечек):

* **CV по ТС**  - GroupKFold по  tr_id  на train+test (39 групп, 5 фолдов,
  3 повтора с разным разбиением). Метрика считается **только по реальным ТС**
  (validate полностью реальный, синтетика  - лишь «объём» для обучения).
* **Holdout**  - обучение на  labels_train , проверка на  labels_test
  (ровно та же ситуация, что и с validate: те же ТС, другие отрезки времени).

Модели предсказывают **остаток**  target − cur_dev_s  (сдвиг относительно
подсказки) с функцией потерь MAE/Huber; итоговый прогноз = cur_dev_s + остаток.

Запуск::

    python -m ml.train --features artifacts/features --out artifacts/model
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from ml.feature_engineering import FEATURE_NAMES

SEED = 42

# Гиперпараметры подобраны по CV (см. README_ML.md, раздел «Подбор»)
CB_PARAMS = dict(loss_function="Huber:delta=80", iterations=1200, learning_rate=0.04, depth=8,
                 l2_leaf_reg=10, random_seed=SEED, verbose=0, thread_count=-1)
LGB_PARAMS = dict(objective="huber", alpha=80.0, learning_rate=0.03, n_estimators=800, num_leaves=31,
                  min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  random_state=SEED, verbose=-1)
XGB_PARAMS = dict(objective="reg:pseudohubererror", huber_slope=80.0, learning_rate=0.03, n_estimators=800,
                  max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
                  random_state=SEED, n_jobs=-1)
ENSEMBLE = ("catboost", "lightgbm", "xgboost")  # финальная модель  - среднее трёх бустингов


# ---------------------------------------------------------------- модели
def make_models() -> dict:
    """Кандидаты: бейзлайны, линейная, леса и три бустинга."""
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "baseline_zero": None,
        "baseline_cur_dev": None,
        "huber_linear": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), HuberRegressor(alpha=1.0, max_iter=500)),
        "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, max_features=0.4, criterion="squared_error",
            n_jobs=-1, random_state=SEED)),
        "extra_trees": lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=SEED)),
        "sk_histgb": lambda: HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.04, max_iter=600, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=SEED),
        "lightgbm": lambda: lgb.LGBMRegressor(**LGB_PARAMS),
        "xgboost": lambda: xgb.XGBRegressor(**XGB_PARAMS),
        "catboost_mae": lambda: cb.CatBoostRegressor(**{**CB_PARAMS, "loss_function": "MAE"}),
        "catboost": lambda: cb.CatBoostRegressor(**CB_PARAMS),
    }


def _fit_predict(name, factory, Xtr, ytr, Xte, wtr=None):
    if name == "baseline_zero":
        return -Xte["cur_dev_s"].to_numpy()  # остаток, дающий итог 0
    if name == "baseline_cur_dev":
        return np.zeros(len(Xte))
    model = factory()
    kw = {}
    if wtr is not None:
        kw = {model.steps[-1][0] + "__sample_weight": wtr} if hasattr(model, "steps") else {"sample_weight": wtr}
    model.fit(Xtr, ytr, **kw)
    return model.predict(Xte)


def score(y, pred, mae_zero=None, mae_base=None):
    """MAE и оценка скора платформы.

     MAE_TARGET  платформы неизвестен; оцениваем его из факта, что baseline
     prediction = cur_dev_s  даёт скор ≈ 0.40.
    """
    mae = float(np.mean(np.abs(y - pred)))
    out = {"mae": round(mae, 2)}
    if mae_zero is not None and mae_base is not None:
        mae_target = mae_zero - (mae_zero - mae_base) / 0.40
        out["score_est"] = round(max(0.0, min(1.0, (mae_zero - mae) / (mae_zero - mae_target))), 3)
    return out


def load(features_dir: Path, mode: str):
    """Прочитать таблицы признаков train / test / validate для режима  hint / stream ."""
    tr = pd.read_csv(features_dir / f"train_{mode}.csv")
    te = pd.read_csv(features_dir / f"test_{mode}.csv")
    va = pd.read_csv(features_dir / f"validate_{mode}.csv")
    return tr, te, va


def _weights(df: pd.DataFrame, syn_weight: float) -> np.ndarray:
    return np.where(df["synthetic"].to_numpy() == 0, 1.0, syn_weight)


def compare(features_dir: Path, mode: str = "hint", syn_weight: float = 1.0, repeats: int = 1) -> pd.DataFrame:
    """CV по ТС + holdout train→test для всех моделей и ансамбля. Возвращает таблицу лидеров."""
    tr, te, _ = load(features_dir, mode)
    full = pd.concat([tr, te], ignore_index=True)
    X, y = full[FEATURE_NAMES], full["target_delay_s"] - full["cur_dev_s"]
    real = full["synthetic"].to_numpy() == 0
    w = _weights(full, syn_weight)
    yt = te["target_delay_s"].to_numpy()
    mz, mb = np.mean(np.abs(yt)), np.mean(np.abs(yt - te["cur_dev_given"].to_numpy()))
    yfull = full["target_delay_s"].to_numpy()

    rows, oofs, holds = [], {}, {}
    for name, fac in make_models().items():
        t0 = time.time()
        cv_mae, oof_sum = [], np.zeros(len(full))
        for rep in range(repeats):
            rng = np.random.RandomState(rep)
            gmap = {g: rng.randint(1 << 30) for g in np.unique(full["tr_id"])}
            groups = full["tr_id"].map(gmap).to_numpy()
            oof = np.zeros(len(full))
            for tri, vai in GroupKFold(n_splits=5).split(X, y, groups):
                oof[vai] = _fit_predict(name, fac, X.iloc[tri], y.iloc[tri], X.iloc[vai], w[tri])
            pred = full["cur_dev_s"].to_numpy() + oof
            cv_mae.append(np.mean(np.abs(yfull[real] - pred[real])))
            oof_sum += pred
        oofs[name] = oof_sum / repeats
        holds[name] = te["cur_dev_s"].to_numpy() + _fit_predict(
            name, fac, tr[FEATURE_NAMES], tr["target_delay_s"] - tr["cur_dev_s"], te[FEATURE_NAMES],
            _weights(tr, syn_weight))
        rows.append({
            "model": name, "cv_mae_real": round(float(np.mean(cv_mae)), 2),
            **{f"holdout_{k}": v for k, v in score(yt, holds[name], mz, mb).items()},
            "fit_sec": round(time.time() - t0, 1),
        })
        print(rows[-1], flush=True)

    ens_oof = np.mean([oofs[m] for m in ENSEMBLE], axis=0)
    ens_hold = np.mean([holds[m] for m in ENSEMBLE], axis=0)
    rows.append({
        "model": "ENSEMBLE(" + "+".join(ENSEMBLE) + ")",
        "cv_mae_real": round(float(np.mean(np.abs(yfull[real] - ens_oof[real]))), 2),
        **{f"holdout_{k}": v for k, v in score(yt, ens_hold, mz, mb).items()},
        "fit_sec": np.nan,
    })
    return pd.DataFrame(rows).sort_values("cv_mae_real")


MODEL_EXT = {"catboost": ".cbm", "lightgbm": ".txt", "xgboost": ".json"}


def save_native(model, name: str, stem: Path) -> Path:
    """Сохранить модель в родном формате библиотеки (переносимо между версиями Python)."""
    path = stem.with_suffix(MODEL_EXT[name])
    if name == "lightgbm":
        model.booster_.save_model(str(path))
    else:
        model.save_model(str(path))
    return path


def fit_final(features_dir: Path, out: Path, syn_weight: float = 1.0) -> dict:
    """Обучить финальные артефакты на train+test.

    Сохраняет для режимов  hint  (есть подсказка cur_dev_s) и  stream
    (только телеметрия + план):

    *  {mode}_reg_catboost.cbm ,  _lightgbm.txt ,  _xgboost.json   - ансамбль регрессоров остатка;
    *  {mode}_q10 / {mode}_q90   - квантили CatBoost (интервал прогноза).
    """
    import catboost as cb

    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    meta = {"features": FEATURE_NAMES, "ensemble": list(ENSEMBLE), "modes": {}}
    for mode in ("hint", "stream"):
        tr, te, _ = load(features_dir, mode)
        full = pd.concat([tr, te], ignore_index=True)
        X = full[FEATURE_NAMES]
        resid = full["target_delay_s"] - full["cur_dev_s"]
        w = _weights(full, syn_weight)
        models = make_models()
        for name in ENSEMBLE:
            m = models[name]()
            m.fit(X, resid, sample_weight=w)
            save_native(m, name, out / f"{mode}_reg_{name}")
        for q in (0.1, 0.9):
            mq = cb.CatBoostRegressor(loss_function=f"Quantile:alpha={q}", iterations=800, learning_rate=0.05,
                                      depth=6, l2_leaf_reg=10, random_seed=SEED, verbose=0)
            mq.fit(X, resid, sample_weight=w)
            mq.save_model(str(out / f"{mode}_q{int(q * 100)}.cbm"))
        meta["modes"][mode] = {"n_train": int(len(full))}
        print("saved", mode, flush=True)
    if "nn" in old:  # секцию нейросети пишет ml.train_nn  - не затираем её
        meta["nn"] = old["nn"]
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["compare", "fit"], help="compare  - лидерборд; fit  - финальные модели")
    ap.add_argument("--features", type=Path, default=Path("artifacts/features"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/model"))
    ap.add_argument("--mode", default="hint", choices=["hint", "stream"])
    ap.add_argument("--syn-weight", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "compare":
        lb = compare(args.features, args.mode, args.syn_weight, args.repeats)
        lb.to_csv(args.out / f"leaderboard_{args.mode}.csv", index=False)
        print(lb.to_string(index=False))
    else:
        fit_final(args.features, args.out, args.syn_weight)


if __name__ == "__main__":
    main()
