"""PyTorch-модель: GRU по последовательности телеметрии + MLP по табличным признакам.

Архитектура ``DelayNet``::

    seq (B, 120, 7) ─► Conv1d(stride 2) ─► GRU(64) ─► h_T ─┐
                                                          ├─► MLP ─► остаток задержки (target − cur_dev_s) / 100
    tab (B, 49) ─► [стандартизация, NaN→0, маска] ─► MLP ─┘

Энкодер последовательности можно предобучить самообучением (``pretext``-голова:
предсказать будущие скорость/путь/стоянки/отход от плана по прошлой телеметрии)
на ТС без разметки, затем дообучить всю сеть на размеченных точках.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.sequence import N_CHANNELS

N_PRETEXT = 5
TARGET_SCALE = 100.0


def device() -> torch.device:
    """GPU, если есть (поощряется ТЗ), иначе CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeqEncoder(nn.Module):
    """Conv1d (свёртка по 4 шагам, шаг 2: 120 → 60 шагов по 30 с) + однослойный GRU.

    Свёртка сжимает последовательность и ловит локальные паттерны (торможение,
    остановка), GRU — динамику за 30 минут. Так обучение на CPU в ~4 раза быстрее,
    чем у двухслойного GRU по полной сетке, без потери качества.
    """

    def __init__(self, c_in: int = N_CHANNELS, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c_in, hidden, kernel_size=4, stride=2, padding=1), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.pretext = nn.Linear(hidden, N_PRETEXT)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        z = self.conv(seq.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(z)
        return h[:, -1]


class DelayNet(nn.Module):
    def __init__(self, n_tab: int, hidden: int = 64, use_tab: bool = True):
        super().__init__()
        self.use_tab = use_tab
        self.enc = SeqEncoder(hidden=hidden)
        tab_out = 0
        if use_tab:
            self.tab = nn.Sequential(nn.Linear(2 * n_tab + 1, 128), nn.GELU(), nn.Dropout(0.2),
                                     nn.Linear(128, 64), nn.GELU())
            tab_out = 64
        else:
            self.tab = nn.Linear(1, 8)  # только cur_dev_s
            tab_out = 8
        self.head = nn.Sequential(nn.Linear(hidden + tab_out, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, seq: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        z = self.enc(seq)
        t = self.tab(tab if self.use_tab else tab[:, -1:])
        return self.head(torch.cat([z, t], dim=1)).squeeze(1)


@dataclass
class TabScaler:
    """Стандартизация табличных признаков; NaN → 0 плюс бинарная маска пропуска."""

    mean: list
    std: list

    @classmethod
    def fit(cls, X: np.ndarray) -> "TabScaler":
        m = np.nanmean(X, axis=0)
        s = np.nanstd(X, axis=0)
        return cls(np.nan_to_num(m).tolist(), np.where(np.nan_to_num(s) > 1e-9, s, 1.0).tolist())

    def transform(self, X: np.ndarray, cur_dev: np.ndarray) -> np.ndarray:
        miss = ~np.isfinite(X)
        Z = np.clip((X - np.asarray(self.mean)) / np.asarray(self.std), -5, 5)
        Z = np.where(miss, 0.0, Z)
        return np.concatenate([Z, miss.astype(float), (cur_dev / 300.0)[:, None]], axis=1).astype(np.float32)


@dataclass
class NNConfig:
    hidden: int = 64
    epochs: int = 60
    lr: float = 2e-3
    batch: int = 128
    weight_decay: float = 1e-4
    huber_delta: float = 1.5   # в единицах TARGET_SCALE → 150 с
    use_tab: bool = True
    seed: int = 0


def pretrain_encoder(seqs: np.ndarray, targets: np.ndarray, hidden: int = 64, epochs: int = 8,
                     seed: int = 0, log=print) -> dict:
    """Самообучение энкодера: по прошлым 30 мин предсказать будущие 5–10 мин движения."""
    torch.manual_seed(seed)
    dev = device()
    enc = SeqEncoder(hidden=hidden).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=2e-3, weight_decay=1e-4)
    S = torch.from_numpy(seqs)
    Y = torch.from_numpy(np.nan_to_num(targets, nan=0.0).astype(np.float32))
    M = torch.from_numpy(np.isfinite(targets).astype(np.float32))
    n = len(S)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            s, y, m = S[b].to(dev), Y[b].to(dev), M[b].to(dev)
            pred = enc.pretext(enc(s))
            loss = (nn.functional.smooth_l1_loss(pred, y, reduction="none", beta=0.5) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        log(f"  pretrain epoch {ep + 1}/{epochs}: loss {tot / n:.4f}")
    return {k: v.cpu() for k, v in enc.state_dict().items()}


def fit_net(seq: np.ndarray, X: np.ndarray, cur_dev: np.ndarray, resid: np.ndarray, cfg: NNConfig,
            weights: np.ndarray | None = None, encoder_state: dict | None = None):
    """Обучить DelayNet на остатке ``target − cur_dev_s``. Возвращает (модель, скейлер)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = device()
    scaler = TabScaler.fit(X)
    tab = scaler.transform(X, cur_dev)
    net = DelayNet(X.shape[1], cfg.hidden, cfg.use_tab).to(dev)
    if encoder_state is not None:
        net.enc.load_state_dict(encoder_state)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(seq) / cfg.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    S, T_ = torch.from_numpy(seq), torch.from_numpy(tab)
    Y = torch.from_numpy((resid / TARGET_SCALE).astype(np.float32))
    W = torch.from_numpy((np.ones(len(seq)) if weights is None else weights).astype(np.float32))
    net.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(S))
        for i in range(0, len(S), cfg.batch):
            b = perm[i:i + cfg.batch]
            pred = net(S[b].to(dev), T_[b].to(dev))
            l = nn.functional.huber_loss(pred, Y[b].to(dev), reduction="none", delta=cfg.huber_delta)
            loss = (l * W[b].to(dev)).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
    net.eval()
    return net, scaler


@torch.no_grad()
def predict_net(net: DelayNet, scaler: TabScaler, seq: np.ndarray, X: np.ndarray, cur_dev: np.ndarray) -> np.ndarray:
    """Прогноз остатка (сек)."""
    dev = next(net.parameters()).device
    tab = torch.from_numpy(scaler.transform(X, cur_dev)).to(dev)
    out = []
    for i in range(0, len(seq), 1024):
        out.append(net(torch.from_numpy(seq[i:i + 1024]).to(dev), tab[i:i + 1024]).cpu().numpy())
    return np.concatenate(out) * TARGET_SCALE if out else np.zeros(0)


def save_net(net: DelayNet, scaler: TabScaler, cfg: NNConfig, n_tab: int, path: Path) -> None:
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "scaler": asdict(scaler),
                "cfg": asdict(cfg), "n_tab": n_tab}, path)


def load_net(path: Path) -> tuple[DelayNet, TabScaler]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = NNConfig(**ck["cfg"])
    net = DelayNet(ck["n_tab"], cfg.hidden, cfg.use_tab)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, TabScaler(**ck["scaler"])


def export_onnx(net: DelayNet, n_tab: int, path: Path) -> None:
    """Экспорт в ONNX (доп. фича «оптимизация инференса»)."""
    from ml.sequence import SEQ_LEN
    net = net.cpu().eval()
    torch.onnx.export(net, (torch.zeros(1, SEQ_LEN, N_CHANNELS), torch.zeros(1, 2 * n_tab + 1)), str(path),
                      input_names=["seq", "tab"], output_names=["resid"],
                      dynamic_axes={"seq": {0: "batch"}, "tab": {0: "batch"}, "resid": {0: "batch"}},
                      dynamo=False)
