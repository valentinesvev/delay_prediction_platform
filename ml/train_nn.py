"""Обучение нейросети (PyTorch) и её смешивание с ансамблем бустингов.

Шаги::

    python -m ml.train_nn build   --data ../dataset      # последовательности + пул для предобучения
    python -m ml.train_nn compare                          # CV/holdout: варианты сети и бленд с бустингами
    python -m ml.train_nn fit                              # финальные веса сети → artifacts/model

Предобучение (self-supervised) идёт **только** на ТС, которых нет в test/validate:
синтетические ТС и реальные ТС без разметки из ``train/traffic.csv``. Цели —
будущее движение по телеметрии, разметка задержек не используется.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from ml.dataset import SPLITS, _to_unix, load_plans, load_tracks
from ml.feature_engineering import FEATURE_NAMES, VehicleTrack, _align
from ml.nn_model import NNConfig, export_onnx, fit_net, predict_net, pretrain_encoder, save_net
from ml.sequence import build_sequence, pretext_targets
from ml.train import ENSEMBLE, load, make_models

SEQ_DIR = Path("artifacts/seq")
N_SEEDS = 3      # сидов в финальной модели
CV_SEEDS = 2     # сидов в CV (ради времени на CPU)


# ---------------------------------------------------------------- данные
def sequences_for(plans: dict, tracks: dict, feats: pd.DataFrame) -> np.ndarray:
    """Последовательности для строк таблицы признаков (dev_ref = её ``cur_dev_s``)."""
    empty = VehicleTrack.from_arrays([], [], [], [], [])
    T = _to_unix(feats["T"])
    return np.stack([build_sequence(plans[int(r)], tracks.get(int(r), empty), float(t), float(c))
                     for r, t, c in zip(feats["tr_id"], T, feats["cur_dev_s"])])


def build(data_dir: Path, features_dir: Path, out: Path = SEQ_DIR, pretrain_step_s: float = 90.0) -> None:
    """Собрать последовательности для размеченных точек и пул для предобучения."""
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    for split, (traffic, schedule, _) in SPLITS.items():
        key = (traffic, schedule)
        if key not in cache:
            cache[key] = (load_plans(data_dir / schedule), load_tracks(data_dir / traffic))
        plans, tracks = cache[key]
        for mode in ("hint", "stream"):
            seqs = sequences_for(plans, tracks, pd.read_csv(features_dir / f"{split}_{mode}.csv"))
            np.save(out / f"{split}_{mode}.npy", seqs)
            print(split, mode, seqs.shape, flush=True)

    # пул предобучения: ТС без меток в test/validate
    plans, tracks = cache[(SPLITS["train"][0], SPLITS["train"][1])]
    labeled_eval = set(pd.read_csv(data_dir / "labels/labels_test.csv")["tr_id"]) | \
        set(pd.read_csv(data_dir / "validate/points.csv")["tr_id"])
    pool = [v for v in tracks if v not in labeled_eval]
    S, Y = [], []
    for v in pool:
        tr, plan = tracks[v], plans.get(v)
        if len(tr.t) < 50:
            continue
        for T in np.arange(tr.t[0] + 1800, tr.t[-1] - 600, pretrain_step_s):
            past = tr.upto(T)
            if len(past.t) == 0 or T - past.t[-1] > 60:
                continue
            ref = _align(plan, past, T, 600, 0.0)[0] if plan is not None else 0.0
            ref = ref if np.isfinite(ref) else 0.0
            S.append(build_sequence(plan, tr, T, ref))
            Y.append(pretext_targets(plan, tr, T, ref))
    S, Y = np.stack(S), np.stack(Y)
    np.save(out / "pretrain_seq.npy", S)
    np.save(out / "pretrain_y.npy", Y)
    print("pretrain pool:", len(pool), "ТС,", S.shape, "окон; без плана:",
          int(sum(1 for v in pool if v not in plans)), "ТС", flush=True)


def _load_seq(split: str, mode: str) -> np.ndarray:
    return np.load(SEQ_DIR / f"{split}_{mode}.npy")


def _encoder(seed: int = 0, epochs: int = 8, log=print) -> dict:
    return pretrain_encoder(np.load(SEQ_DIR / "pretrain_seq.npy"), np.load(SEQ_DIR / "pretrain_y.npy"),
                            epochs=epochs, seed=seed, log=log)


def _fit_pred(seq_tr, df_tr, seq_te, df_te, cfg: NNConfig, enc, seeds=N_SEEDS) -> np.ndarray:
    X, Xt = df_tr[FEATURE_NAMES].to_numpy(float), df_te[FEATURE_NAMES].to_numpy(float)
    c, ct = df_tr["cur_dev_s"].to_numpy(float), df_te["cur_dev_s"].to_numpy(float)
    r = df_tr["target_delay_s"].to_numpy(float) - c
    preds = []
    for s in range(seeds):
        cfg_s = NNConfig(**{**cfg.__dict__, "seed": s})
        net, sc = fit_net(seq_tr, X, c, r, cfg_s, encoder_state=enc)
        preds.append(predict_net(net, sc, seq_te, Xt, ct))
    return ct + np.mean(preds, axis=0)


# ---------------------------------------------------------------- сравнение
def compare(mode: str = "hint", only_best: bool = False) -> pd.DataFrame:
    """CV по ТС (те же фолды для сети и бустингов) + holdout; подбор веса бленда."""
    tr, te, _ = load(Path("artifacts/features"), mode)
    s_tr, s_te = _load_seq("train", mode), _load_seq("test", mode)
    full = pd.concat([tr, te], ignore_index=True)
    s_full = np.concatenate([s_tr, s_te])
    real = full["synthetic"].to_numpy() == 0
    y = full["target_delay_s"].to_numpy()
    yt = te["target_delay_s"].to_numpy()
    rng = np.random.RandomState(0)
    gmap = {g: rng.randint(1 << 30) for g in np.unique(full["tr_id"])}
    folds = list(GroupKFold(5).split(full, y, full["tr_id"].map(gmap)))

    t0 = time.time()
    enc = _encoder(log=lambda *_: None)
    print(f"encoder pretrained in {time.time() - t0:.0f}s", flush=True)

    variants = {
        "gru_seq_only+pretrain": (NNConfig(use_tab=False), enc),
        "gru+tab": (NNConfig(), None),
        "gru+tab+pretrain": (NNConfig(), enc),
    }
    if only_best:
        variants = {"gru+tab+pretrain": variants["gru+tab+pretrain"]}
    rows, oof, hold = [], {}, {}
    for name, (cfg, e) in variants.items():
        t0 = time.time()
        o = np.zeros(len(full))
        for a, b in folds:
            o[b] = _fit_pred(s_full[a], full.iloc[a], s_full[b], full.iloc[b], cfg, e, CV_SEEDS)
        oof[name] = o
        hold[name] = _fit_pred(s_tr, tr, s_te, te, cfg, e, CV_SEEDS)
        rows.append({"model": name, "cv_mae_real": round(float(np.mean(np.abs(y - o)[real])), 2),
                     "holdout_mae": round(float(np.mean(np.abs(yt - hold[name]))), 2),
                     "fit_sec": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)

    # бустинги на тех же фолдах
    models = make_models()
    X, r = full[FEATURE_NAMES], y - full["cur_dev_s"].to_numpy()
    gb = np.zeros(len(full))
    for a, b in folds:
        gb[b] = full["cur_dev_s"].to_numpy()[b] + np.mean(
            [models[n]().fit(X.iloc[a], r[a]).predict(X.iloc[b]) for n in ENSEMBLE], axis=0)
    gb_hold = te["cur_dev_s"].to_numpy() + np.mean(
        [models[n]().fit(tr[FEATURE_NAMES], tr["target_delay_s"] - tr["cur_dev_s"]).predict(te[FEATURE_NAMES])
         for n in ENSEMBLE], axis=0)
    rows.append({"model": "boosting_ensemble", "cv_mae_real": round(float(np.mean(np.abs(y - gb)[real])), 2),
                 "holdout_mae": round(float(np.mean(np.abs(yt - gb_hold))), 2), "fit_sec": np.nan})

    best_nn = min(variants, key=lambda k: np.mean(np.abs(y - oof[k])[real]))
    grid = np.round(np.arange(0, 1.01, 0.05), 2)
    cv_w = [np.mean(np.abs(y - ((1 - w) * gb + w * oof[best_nn]))[real]) for w in grid]
    w = float(grid[int(np.argmin(cv_w))])
    blend_h = (1 - w) * gb_hold + w * hold[best_nn]
    rows.append({"model": f"BLEND boosting + {best_nn} (w_nn={w})", "cv_mae_real": round(float(min(cv_w)), 2),
                 "holdout_mae": round(float(np.mean(np.abs(yt - blend_h))), 2), "fit_sec": np.nan})
    lb = pd.DataFrame(rows)
    print(lb.to_string(index=False))
    Path("artifacts/model").mkdir(parents=True, exist_ok=True)
    lb.to_csv(f"artifacts/model/leaderboard_nn_{mode}.csv", index=False)
    json.dump({"best_nn": best_nn, "w_nn": w, "cv_curve": dict(zip(map(str, grid), map(float, cv_w)))},
              open(f"artifacts/model/blend_{mode}.json", "w"), indent=1)
    return lb


# ---------------------------------------------------------------- финальное обучение
def fit(out: Path = Path("artifacts/model")) -> None:
    """Предобучить энкодер и обучить сеть (N_SEEDS сидов) на train+test для hint и stream."""
    t0 = time.time()
    enc = _encoder()
    print(f"encoder: {time.time() - t0:.0f}s", flush=True)
    torch.save(enc, out / "nn_encoder_pretrained.pt")
    meta = json.loads((out / "meta.json").read_text())
    meta["nn"] = {}
    for mode in ("hint", "stream"):
        blend = json.loads((out / f"blend_{mode}.json").read_text())
        use_pre = "pretrain" in blend["best_nn"]
        cfg = NNConfig(use_tab="tab" in blend["best_nn"])
        tr, te, _ = load(Path("artifacts/features"), mode)
        full = pd.concat([tr, te], ignore_index=True)
        seq = np.concatenate([_load_seq("train", mode), _load_seq("test", mode)])
        X, c = full[FEATURE_NAMES].to_numpy(float), full["cur_dev_s"].to_numpy(float)
        r = full["target_delay_s"].to_numpy(float) - c
        files = []
        for s in range(N_SEEDS):
            net, sc = fit_net(seq, X, c, r, NNConfig(**{**cfg.__dict__, "seed": s}),
                              encoder_state=enc if use_pre else None)
            p = out / f"{mode}_nn_seed{s}.pt"
            save_net(net, sc, NNConfig(**{**cfg.__dict__, "seed": s}), len(FEATURE_NAMES), p)
            files.append(p.name)
            if s == 0:
                try:
                    export_onnx(net, len(FEATURE_NAMES), out / f"{mode}_nn.onnx")
                except Exception as e:  # ONNX — опционально
                    print("onnx export skipped:", e)
        meta["nn"][mode] = {"files": files, "variant": blend["best_nn"], "w_nn": blend["w_nn"]}
        print("saved nn", mode, meta["nn"][mode], flush=True)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["build", "compare", "fit"])
    ap.add_argument("--data", type=Path, default=Path("../dataset"))
    ap.add_argument("--mode", default="hint", choices=["hint", "stream"])
    ap.add_argument("--only-best", action="store_true", help="сравнивать только gru+tab+pretrain")
    args = ap.parse_args()
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    if args.command == "build":
        build(args.data, Path("artifacts/features"))
    elif args.command == "compare":
        compare(args.mode, args.only_best)
    else:
        fit()


if __name__ == "__main__":
    main()
