"""
plot_ew_pipelines_extra.py
===========================
Два дополнительных пайплайна, аналогичных plot_ew_hybrid_full.py:

  1. no_vel         — признаки последних результатов БЕЗ скорости КВМ.
  2. cme_angles     — добавлены позиционный угол и угол раствора КВМ
                      к параметрам КВМ (скорость остаётся).

Для каждого пайплайна строим два варианта:
    - weighted    (гибридные веса: West=Density, East=Target α=1.5)
    - unweighted  (без перевзвешивания, w=1)

Результаты:
    results_ew_no_vel/{weighted,unweighted}/plots/
    results_ew_cme_angles/{weighted,unweighted}/plots/

Запуск: python plot_ew_pipelines_extra.py
"""

import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import plot_ew_hybrid_full as P

ROOT = Path(__file__).parent


FS_NO_VEL = [
    ("Базовая",           ["helio_lon", "log_goes_peak_flux"]),
    ("Флюэс вместо пика", ["helio_lon", "log_fluence"]),
    ("Обе координаты",    ["helio_lon", "helio_lat", "log_goes_peak_flux"]),
    ("Координаты+флюэс",  ["helio_lon", "helio_lat", "log_fluence"]),
]

FS_CME_ANGLES = [
    ("Базовая",           ["helio_lon", "log_goes_peak_flux",
                           "log_cme_velocity", "cme_pa_deg", "cme_width_deg"]),
    ("Флюэс вместо пика", ["helio_lon", "log_fluence",
                           "log_cme_velocity", "cme_pa_deg", "cme_width_deg"]),
    ("Обе координаты",    ["helio_lon", "helio_lat", "log_goes_peak_flux",
                           "log_cme_velocity", "cme_pa_deg", "cme_width_deg"]),
    ("Координаты+флюэс",  ["helio_lon", "helio_lat", "log_fluence",
                           "log_cme_velocity", "cme_pa_deg", "cme_width_deg"]),
]


CONFIGS = [
    ("no_vel",     "weighted",   FS_NO_VEL,     True),
    ("no_vel",     "unweighted", FS_NO_VEL,     False),
    ("cme_angles", "weighted",   FS_CME_ANGLES, True),
    ("cme_angles", "unweighted", FS_CME_ANGLES, False),
]


def _ones_group_weights(group, X_tr_s, X_te_s, jmax_values):
    return np.ones(len(X_tr_s))


def _ones_combined_weights(train_west, train_east, X_tr_s_all, X_te_s):
    return np.ones(len(X_tr_s_all))


def run_one(feat_sets, out_dir: Path, weighted: bool, approach_label: str):
    importlib.reload(P)  # сбросить любые патчи и перечитать исходники
    # Переопределяем нужные глобалы ПОСЛЕ reload
    P.FEATURE_SETS = feat_sets
    P.PLOTS_DIR = out_dir
    P.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not weighted:
        P.get_weights = _ones_group_weights
        P._combined_weights = _ones_combined_weights
        P.GROUP_APPROACH = {"West": "Без весов", "East": "Без весов"}
    else:
        P.GROUP_APPROACH = {"West": approach_label + " · Density-W",
                            "East": approach_label + " · Target-W (α=1.5)"}

    print(f"\n========== {out_dir} ==========")
    P.main()


def main():
    for pipeline, mode, feat_sets, weighted in CONFIGS:
        out_dir = ROOT / f"results_ew_{pipeline}" / mode / "plots"
        run_one(feat_sets, out_dir, weighted, approach_label=pipeline)

    print("\nВсё готово. Сгенерированы 4 набора графиков.")


if __name__ == "__main__":
    main()
