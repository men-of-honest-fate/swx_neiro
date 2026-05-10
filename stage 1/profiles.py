"""
Построение и контроль качества профилей фазы нарастания СПС.

Для каждого события из каталога:
  - срез [onset, peak] из протонного ряда,
  - log10(flux), Δlog10,
  - интерполяция коротких пропусков,
  - отсев событий с >20% пропусков и фазой >20 ч,
  - флаг «составное» (по второй ≥M-вспышке в окне; см. ограничения ниже).

Сохраняет parquet в `stage 1/results/profiles/{event_id}.parquet`
и сводный `profiles_index.parquet` со всеми метаданными.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
PROFILE_DIR  = ROOT / "results" / "profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR    = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from catalog import load_catalog          # noqa: E402
from data    import load_protons_window   # noqa: E402

CADENCE_MIN     = 5
MAX_GAP_FRAC    = 0.20
MAX_INTERP_GAP  = 2
MAX_RISE_HOURS  = 20.0
LOG_FLOOR_PFU   = 1e-3       # min для log10 (защищает от нулей)


def _resample_5min(sub: pd.DataFrame, onset, peak) -> pd.DataFrame:
    """Привести к равномерной 5-мин сетке [onset, peak] включительно."""
    grid = pd.date_range(onset.floor(f"{CADENCE_MIN}min"),
                         peak.ceil(f"{CADENCE_MIN}min"),
                         freq=f"{CADENCE_MIN}min")
    out = sub["flux"].reindex(grid, method="nearest", tolerance=pd.Timedelta(f"{CADENCE_MIN}min"))
    return out.to_frame("flux")


def _interpolate_short_gaps(s: pd.Series, max_gap: int) -> tuple[pd.Series, float]:
    """Линейная интерполяция пропусков длиной ≤ max_gap. Возвращает (s, gap_frac до интерп.)."""
    is_nan = s.isna().to_numpy()
    gap_frac = float(is_nan.mean())
    if gap_frac == 0:
        return s, 0.0
    s_interp = s.interpolate(method="time", limit=max_gap, limit_area="inside")
    return s_interp, gap_frac


def build_profile(row: pd.Series) -> dict | None:
    """Построить профиль одного события. None, если событие отсеяно."""
    onset, peak = row["onset_dt"], row["peak_dt"]
    rise_h = (peak - onset).total_seconds() / 3600.0
    if rise_h > MAX_RISE_HOURS:
        return {"event_id": row["event_id"], "status": "rejected_rise_too_long",
                "rise_hours": rise_h}

    raw = load_protons_window(onset, peak, pad_min=0)
    if raw is None or raw.empty:
        return {"event_id": row["event_id"], "status": "rejected_no_data"}

    ts = _resample_5min(raw, onset, peak)
    ts["flux"], gap_before = _interpolate_short_gaps(ts["flux"], MAX_INTERP_GAP)
    gap_after = float(ts["flux"].isna().mean())

    if gap_after > MAX_GAP_FRAC:
        return {"event_id": row["event_id"], "status": "rejected_too_many_gaps",
                "gap_frac": gap_after, "gap_frac_raw": gap_before}

    flux = ts["flux"].clip(lower=LOG_FLOOR_PFU).to_numpy()
    log_J     = np.log10(flux)
    delta_log = np.concatenate([[0.0], np.diff(log_J)])

    profile = pd.DataFrame({
        "timestamp":   ts.index,
        "flux":        ts["flux"].to_numpy(),
        "log_J":       log_J,
        "delta_log_J": delta_log,
    })

    out_path = PROFILE_DIR / f"{row['event_id']}.parquet"
    profile.to_parquet(out_path, index=False)

    return {
        "event_id":     row["event_id"],
        "status":       "ok",
        "n_steps":      len(profile),
        "rise_hours":   rise_h,
        "gap_frac_raw": gap_before,
        "gap_frac":     gap_after,
        "log_J_max":    float(log_J.max()),
        "log_J_real":   float(np.log10(max(row["Jmax"], LOG_FLOOR_PFU))),
        "path":         str(out_path.relative_to(PROJECT_ROOT)),
    }


def build_all() -> pd.DataFrame:
    cat = load_catalog()
    print(f"Каталог: {len(cat)} событий")
    rows = []
    for i, (_, row) in enumerate(cat.iterrows(), 1):
        if i % 25 == 1:
            print(f"  [{i:3d}/{len(cat)}] {row['event_id']}")
        rec = build_profile(row)
        if rec is None:
            continue
        # Скопировать важные метаданные каталога в индекс
        for k in ("Цикл", "split", "group", "onset_dt", "peak_dt",
                  "Jmax", "T_delta", "helio_lon"):
            rec[k] = row[k]
        rows.append(rec)

    idx = pd.DataFrame(rows)
    idx.to_parquet(ROOT / "results" / "profiles_index.parquet", index=False)

    summary = idx.groupby("status").size()
    print("\nСтатусы:")
    print(summary.to_string())
    n_ok = int(summary.get("ok", 0))
    print(f"\n✓ Готовых профилей: {n_ok}")
    return idx


def plot_qc(idx: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = idx[idx["status"] == "ok"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(ok["n_steps"], bins=40, color="#2c7fb8")
    axes[0].axvline(ok["n_steps"].median(), color="red", ls="--", label=f"med={ok['n_steps'].median():.0f}")
    axes[0].set_xlabel("Длина профиля (5-мин шаги)"); axes[0].set_ylabel("Кол-во"); axes[0].legend()

    axes[1].hist(ok["rise_hours"], bins=30, color="#41b6c4")
    axes[1].set_xlabel("Фаза нарастания (ч)"); axes[1].set_ylabel("Кол-во")

    axes[2].hist(ok["gap_frac"] * 100, bins=20, color="#7fcdbb")
    axes[2].set_xlabel("Доля пропусков (%, после интерп.)"); axes[2].set_ylabel("Кол-во")

    fig.suptitle(f"Контроль качества профилей  (n_ok = {len(ok)})", y=1.02)
    fig.tight_layout()
    out = PLOTS_DIR / "profile_qc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"График сохранён: {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        idx = build_all()
        plot_qc(idx)
    elif cmd == "qc":
        idx = pd.read_parquet(ROOT / "results" / "profiles_index.parquet")
        plot_qc(idx)
    else:
        print("Использование: python profiles.py [build|qc]")
