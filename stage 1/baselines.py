"""
Бейзлайны для сравнения с LSTM-наукастером.

Все три имеют единый интерфейс:
    predict(observed_log_J, k, N_max, prior_J=None, prior_T=None) -> np.ndarray
        observed_log_J : [k] — наблюдённые log10 значений до момента t_now
        k              : длина наблюдённой части (шаги)
        N_max          : длина прогноза (шаги, считая от шага k)
        prior_*        : нужны только prior-only (для остальных могут быть None)

Возвращают: np.ndarray[N_max-k] — прогноз log10 J на оставшиеся шаги.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

CADENCE_MIN = 5
LOG_FLOOR   = -3.0  # log10(1e-3 pfu) — соответствует LOG_FLOOR_PFU в profiles.py


# ── 6a. Persistence ──────────────────────────────────────────────────────────

def persistence_predict(observed_log_J: np.ndarray, k: int, N_max: int,
                         prior_J: float | None = None,
                         prior_T: float | None = None) -> np.ndarray:
    if k == 0:
        return np.full(N_max, LOG_FLOOR)
    last = float(observed_log_J[-1])
    return np.full(N_max - k, last)


# ── 6b. Степенной фит ────────────────────────────────────────────────────────

def _power_law(t, log_J0, alpha):
    return log_J0 + alpha * np.log(np.maximum(t, 1e-3))


def power_law_predict(observed_log_J: np.ndarray, k: int, N_max: int,
                       prior_J: float | None = None,
                       prior_T: float | None = None) -> np.ndarray:
    if k < 2:
        return np.full(N_max - k, observed_log_J[-1] if k > 0 else LOG_FLOOR)

    t = np.arange(1, k + 1, dtype=float)   # шаги от onset (1, 2, ...)
    try:
        popt, _ = curve_fit(_power_law, t, observed_log_J,
                             p0=[float(observed_log_J[0]), 0.5], maxfev=2000)
    except Exception:
        return np.full(N_max - k, float(observed_log_J[-1]))

    t_future = np.arange(k + 1, N_max + 1, dtype=float)
    pred     = _power_law(t_future, *popt)
    return np.clip(pred, LOG_FLOOR, 5.0)   # ≤ 10⁵ pfu — sanity


# ── 6c. Prior-only ───────────────────────────────────────────────────────────

def _gamma_profile(t_steps: np.ndarray, peak_step: float, log_peak: float,
                   shape: float = 2.5) -> np.ndarray:
    """Нормированный гамма-подобный профиль:
       форма ~ (t/peak)^(shape-1) * exp(shape · (1 - t/peak))
    Это даёт max в t=peak с f=1, монотонный рост до пика и спад после.
    """
    eps = 1e-3
    x = np.maximum(t_steps, eps) / max(peak_step, eps)
    f = (x ** (shape - 1)) * np.exp(shape * (1 - x))
    f = np.maximum(f, 1e-6)
    return log_peak + np.log10(f)   # log10 от формы (1 на пике → 0 в логе → log_peak)


def prior_only_predict(observed_log_J: np.ndarray, k: int, N_max: int,
                        prior_J: float | None = None,
                        prior_T: float | None = None) -> np.ndarray:
    """Параметризованная гамма-кривая с пиком в (t_onset + T_delta_prior, J_max_prior).

    prior_J — log10(J_max_prior) ! на вход подаём уже в log-space
    prior_T — T_delta_prior в часах
    """
    if prior_J is None or prior_T is None or np.isnan(prior_J) or np.isnan(prior_T):
        return persistence_predict(observed_log_J, k, N_max)

    peak_step = max(prior_T * 60.0 / CADENCE_MIN, 1.0)
    t_future  = np.arange(k + 1, N_max + 1, dtype=float)
    pred      = _gamma_profile(t_future, peak_step=peak_step, log_peak=float(prior_J))
    return np.clip(pred, LOG_FLOOR, 5.0)


# ── Реестр для evaluate.py ───────────────────────────────────────────────────

BASELINES = {
    "persistence": persistence_predict,
    "power_law":   power_law_predict,
    "prior_only":  prior_only_predict,
}


# ── Smoke ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    rng = np.random.default_rng(0)
    L = 60
    t = np.arange(1, L + 1)
    true = -1 + 2 * np.log10(t / 30) + 0.5 * np.exp(-((t - 30) / 8) ** 2) + rng.normal(0, 0.05, L)
    true = np.clip(true, -3, 3)

    k = 15
    obs = true[:k]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, true, "k-", lw=1.5, label="истинный профиль")
    ax.axvline(k, color="grey", ls=":")

    for name, fn in BASELINES.items():
        pred = fn(obs, k, L, prior_J=2.0, prior_T=2.5)
        ax.plot(np.arange(k + 1, L + 1), pred, "--", label=name)

    ax.legend(); ax.set_xlabel("шаг (5 мин)"); ax.set_ylabel("log10 J")
    ax.set_title("Baselines smoke")
    out = Path(__file__).parent / "results" / "plots" / "baselines_smoke.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"График сохранён: {out}")
