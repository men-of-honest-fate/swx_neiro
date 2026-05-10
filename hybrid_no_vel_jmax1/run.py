"""
hybrid_no_vel_jmax1/run.py
==========================
Гибридные W/E-модели (West=Density-W, East=Target-W α=1.5) на наборах
признаков БЕЗ скорости КВМ, обученные на всех СПС с J_max ≥ 1 pfu
(вместо ≥ 10 pfu, как в stage 0).

Результаты: hybrid_no_vel_jmax1/plots/

Запуск: python hybrid_no_vel_jmax1/run.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
STAGE0       = PROJECT_ROOT / "stage 0"

sys.path.insert(0, str(STAGE0))
sys.path.insert(0, str(PROJECT_ROOT))

import plot_ew_hybrid_full as P
from spe_utils import build_features, COL_CYCLE


JMAX_MIN = 1.0  # вместо 10 в stage 0


def target_weights_clip1(jmax_values: np.ndarray) -> np.ndarray:
    """Target-weights с нижним clip = JMAX_MIN (вместо 10 как в stage 0).
    Так событиям 1–10 pfu тоже даётся плавно растущий вес."""
    y_log = np.log10(np.clip(jmax_values, JMAX_MIN, None))
    raw   = (y_log - y_log.min() + 0.5) ** P.ALPHA_TW
    return raw / raw.mean()

FS_NO_VEL = [
    ("Базовая",           ["helio_lon", "log_goes_peak_flux"]),
    ("Флюэс вместо пика", ["helio_lon", "log_fluence"]),
    ("Обе координаты",    ["helio_lon", "helio_lat", "log_goes_peak_flux"]),
    ("Координаты+флюэс",  ["helio_lon", "helio_lat", "log_fluence"]),
]


def load_splits_jmax1():
    df = build_features(
        pd.read_excel(PROJECT_ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
                      sheet_name="Флюэс GOES")
    )
    cycle     = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    tdelta    = pd.to_numeric(df["T_delta"],       errors="coerce")
    goes_rise = pd.to_numeric(df["goes_rise_min"], errors="coerce")
    mask = (
        (df["Jmax"].fillna(0) >= JMAX_MIN) &
        (tdelta.fillna(0) <= 40) &
        (goes_rise.fillna(0) <= 120)
    )
    full   = df[mask].copy()
    tr_all = full[cycle.isin([23, 24])].copy()
    te_all = full[cycle.isin([25])].copy()
    splits = {
        "West": (tr_all[tr_all["helio_lon"] > 0].copy(),
                 te_all[te_all["helio_lon"]  > 0].copy()),
        "East": (tr_all[tr_all["helio_lon"] < 0].copy(),
                 te_all[te_all["helio_lon"]  < 0].copy()),
    }
    print(f"Порог J_max ≥ {JMAX_MIN} pfu")
    for g, (tr, te) in splits.items():
        print(f"  {g}: Train={len(tr)}  Test={len(te)}")
    return splits


def main():
    P.load_splits     = load_splits_jmax1
    P._target_weights = target_weights_clip1   # clip 10 → 1
    P.FEATURE_SETS    = FS_NO_VEL
    P.PLOTS_DIR       = ROOT / "plots"
    P.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"== Hybrid W/E · no_vel · J_max ≥ {JMAX_MIN} pfu ==")
    print(f"Результаты: {P.PLOTS_DIR}")
    P.main()


if __name__ == "__main__":
    main()
