"""
Метрики и визуализация для LSTM-наукастера и бейзлайнов.

Метрики (PLAN.md §7), считаются для каждой пары (event, k):
  - RMSE(log J)        — по остатку профиля
  - ΔJ_max             — |max(pred) − log10 J_max_real|
  - ΔT_peak            — |t_pred_peak − t_real_peak| в часах

Главный график: ΔJ_max vs шаг от onset, по предикторам, медианы по test (SC25).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from baselines import BASELINES, CADENCE_MIN, LOG_FLOOR
from dataset   import (PriorNormalizer, load_index, load_profile_array,
                       K_MIN, N_MAX)
from model     import LSTMNowcaster

PLOTS_DIR  = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Загрузка чекпоинта ───────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> LSTMNowcaster:
    state = torch.load(ckpt_path, map_location=device)
    cfg   = state["config"]
    model = LSTMNowcaster(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.0),
        n_max=cfg["n_max"],
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


# ── Прогнозы ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def lstm_predict(model: LSTMNowcaster, profile: np.ndarray,
                  k: int, prior_norm: np.ndarray, device: torch.device,
                  n_max: int) -> np.ndarray:
    """Прогноз LSTM для шагов [k, n_max). Возвращает массив длины (n_max - k)."""
    enc      = torch.from_numpy(profile[:k]).unsqueeze(0).to(device)             # [1, k, 2]
    enc_mask = torch.ones(1, k, device=device)
    prior    = torch.from_numpy(prior_norm).unsqueeze(0).to(device)              # [1, 2]
    preds    = model(enc, enc_mask, prior).cpu().numpy().reshape(-1)             # [n_max]
    return preds[k:]


def evaluate_event(profile: np.ndarray, log_J_max_real: float,
                   peak_step_real: int, prior_J: float, prior_T: float,
                   prior_norm: np.ndarray,
                   model: LSTMNowcaster | None,
                   device: torch.device,
                   k_min: int, n_max: int) -> list[dict]:
    """Вернуть список dict-метрик для (k, predictor)."""
    L = len(profile)
    log_J = profile[:, 0]
    rows  = []

    for k in range(k_min, L):
        obs = log_J[:k]
        true_future = log_J[k:]
        T_future    = len(true_future)

        # 1) бейзлайны
        for name, fn in BASELINES.items():
            pred_full = fn(obs, k, n_max, prior_J=prior_J, prior_T=prior_T)
            pred = pred_full[:T_future]
            rows.append(_metric_row(name, k, pred, true_future,
                                     log_J_max_real, peak_step_real, k_offset=k))

        # 2) LSTM
        if model is not None:
            pred_full = lstm_predict(model, profile, k, prior_norm, device, n_max)
            pred = pred_full[:T_future]
            rows.append(_metric_row("lstm", k, pred, true_future,
                                     log_J_max_real, peak_step_real, k_offset=k))

    return rows


def _metric_row(name: str, k: int, pred: np.ndarray, true_future: np.ndarray,
                 log_J_max_real: float, peak_step_real: int,
                 k_offset: int) -> dict:
    if len(pred) == 0 or len(true_future) == 0:
        return {"predictor": name, "k": k, "rmse_log_J": np.nan,
                "delta_J_max": np.nan, "delta_T_peak_h": np.nan}

    rmse = float(np.sqrt(np.mean((pred - true_future) ** 2)))
    log_J_pred_max = float(pred.max())
    delta_jmax = abs(log_J_pred_max - log_J_max_real)

    # Время пика прогноза в шагах от onset
    pred_peak_step = k_offset + int(np.argmax(pred)) + 1   # +1 чтобы шаг считался от onset
    delta_t_peak_h = abs(pred_peak_step - peak_step_real) * CADENCE_MIN / 60.0

    return {
        "predictor":      name,
        "k":              k,
        "rmse_log_J":     rmse,
        "delta_J_max":    delta_jmax,
        "delta_T_peak_h": delta_t_peak_h,
    }


# ── Прогон по тестовому набору ───────────────────────────────────────────────

def evaluate_split(split: str = "test", model_ckpt: Path | None = None,
                   k_min: int = K_MIN, n_max: int = N_MAX) -> pd.DataFrame:
    idx = load_index()
    idx = idx.dropna(subset=["J_max_prior", "T_delta_prior"]).reset_index(drop=True)
    sub = idx[idx["split"] == split].reset_index(drop=True)
    print(f"Оценка на split={split}: {len(sub)} событий")

    normalizer = PriorNormalizer().fit(idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_ckpt, device) if model_ckpt is not None else None

    all_rows = []
    for _, row in sub.iterrows():
        profile = load_profile_array(row["event_id"])
        if len(profile) > n_max:
            profile = profile[:n_max]
        log_jmax = float(row["log_J_real"])
        peak_step_real = int(np.argmax(profile[:, 0])) + 1
        prior_J = float(row["J_max_prior"])
        prior_T = float(row["T_delta_prior"])
        prior_norm = np.array(normalizer.transform(prior_J, prior_T), dtype=np.float32)

        rows = evaluate_event(profile, log_jmax, peak_step_real,
                              prior_J, prior_T, prior_norm,
                              model, device, k_min, n_max)
        for r in rows:
            r["event_id"] = row["event_id"]
            r["group"]    = row["group"]
            r["L"]        = len(profile)
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    out = ROOT / "results" / f"metrics_{split}.parquet"
    df.to_parquet(out, index=False)
    print(f"✓ Метрики: {out}  ({len(df)} строк)")
    return df


# ── Сводные таблицы и графики ────────────────────────────────────────────────

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Сводная по предиктору × группе × таргету."""
    agg = (df.groupby(["predictor", "group"])
             .agg(rmse_log_J_mean      = ("rmse_log_J",     "mean"),
                  delta_J_max_median    = ("delta_J_max",    "median"),
                  delta_J_max_p90       = ("delta_J_max",    lambda s: s.quantile(0.9)),
                  share_lt_1_order      = ("delta_J_max",    lambda s: float((s < 1.0).mean())),
                  delta_T_peak_h_median = ("delta_T_peak_h", "median"))
             .reset_index())
    return agg


def plot_delta_jmax_vs_time(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"persistence": "#888888", "power_law": "#ff7f0e",
              "prior_only":  "#1f77b4", "lstm":      "#d62728"}

    for predictor in df["predictor"].unique():
        sub = df[df["predictor"] == predictor]
        if sub.empty:
            continue
        agg = sub.groupby("k")["delta_J_max"].median().reset_index()
        t_h = agg["k"] * CADENCE_MIN / 60.0
        ax.plot(t_h, agg["delta_J_max"],
                color=colors.get(predictor, "k"), lw=2, label=predictor)

    ax.set_xlabel("Время после onset, ч")
    ax.set_ylabel("Медиана |Δ log10 J_max|")
    ax.set_title("Ошибка прогноза пика vs длина окна наблюдений")
    ax.axhline(1.0, color="grey", ls=":", lw=1, label="1 порядок")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"График: {out_path}")


def plot_case_studies(df: pd.DataFrame, n: int = 4) -> None:
    """Несколько event-level графиков: профиль и прогнозы LSTM в разные k."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_events = sorted(df["event_id"].unique())[:n]
    if not test_events:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, eid in zip(axes, test_events):
        profile = load_profile_array(eid)
        t_h = np.arange(1, len(profile) + 1) * CADENCE_MIN / 60.0
        ax.plot(t_h, profile[:, 0], "k-", lw=1.5, label="истинный")
        ax.set_title(eid); ax.set_xlabel("ч после onset")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("log10 J (pfu)")
    out = PLOTS_DIR / "case_studies.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Case studies: {out}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "baselines"
    if cmd == "baselines":
        df = evaluate_split("test", model_ckpt=None)
        print("\n== Сводка ==")
        print(summary_table(df).to_string(index=False))
        plot_delta_jmax_vs_time(df, PLOTS_DIR / "delta_jmax_vs_time_baselines.png")

    elif cmd == "full":
        ckpt = Path(sys.argv[2]) if len(sys.argv) > 2 else \
               ROOT / "results" / "checkpoints" / "final" / "final.pt"
        df = evaluate_split("test", model_ckpt=ckpt)
        print("\n== Сводка ==")
        print(summary_table(df).to_string(index=False))
        plot_delta_jmax_vs_time(df, PLOTS_DIR / "delta_jmax_vs_time.png")
        plot_case_studies(df)

    else:
        print("Использование: python evaluate.py [baselines|full [ckpt]]")
