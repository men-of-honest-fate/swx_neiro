"""
Подбор гиперпараметров (RandomizedSearchCV) для двух наборов признаков:
  - baseline : helio_lon, log_flare_power, log_cme_velocity
  - best     : helio_lon, helio_lat, log_flare_power, log_cme_velocity

Для каждого набора и каждого таргета (Jmax, T_delta) обучаются все модели
с подбором гиперпараметров, выводятся финальные метрики и сохраняются графики.

Запуск: python tune.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spe_utils import (
    build_features, fit_and_score, compute_importances,
    plot_importances, plot_scatter, plot_residuals,
    compute_metrics, COL_CYCLE,
)

ROOT       = Path(__file__).parent.parent.parent
PIPELINE   = Path(__file__).parent
INPUT_XLSX = ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx"
SHEET      = "Флюэс GOES"
PLOTS_DIR  = PIPELINE / "plots" / "tuned"

CONFIGS = {
    "baseline": {
        "label":    "Базовая (lon, log_power, log_CME)",
        "features": ["helio_lon", "log_flare_power", "log_cme_velocity"],
        "f_labels": ["Гелиодолгота", "log10(интенс.)", "log10(КВМ)"],
        "subdir":   "baseline",
    },
    "best": {
        "label":    "Обе координаты (lon, lat, log_power, log_CME)",
        "features": ["helio_lon", "helio_lat", "log_flare_power", "log_cme_velocity"],
        "f_labels": ["Гелиодолгота", "Гелиоширота", "log10(интенс.)", "log10(КВМ)"],
        "subdir":   "best",
    },
}

TARGETS = [
    dict(col="Jmax",    label="Jmax (pfu)",      log=True),
    dict(col="T_delta", label="T_delta (часы)",  log=False),
]


def _print_metrics(name, m):
    parts = [f"RMSE={m['RMSE']:.2f}"]
    if "RMSLE_log10" in m:
        parts += [f"RMSLE_log10={m['RMSLE_log10']:.3f}",
                  f"R2_log={m['R2_log']:.3f}",
                  f"Spearman={m['Spearman']:.3f}"]
    else:
        parts += [f"R2={m.get('R2',0):.3f}",
                  f"Spearman={m.get('Spearman',0):.3f}",
                  f"MedAE={m.get('MedAE',0):.2f}"]
    print(f"    {name:<12} " + "  ".join(parts))


def run_config(cfg_name, cfg, train_df, test_df):
    print(f"\n{'='*65}")
    print(f"Конфигурация: {cfg['label']}")
    print("="*65)

    results = {}
    for tgt in TARGETS:
        print(f"\n  -- Цель: {tgt['label']} --")
        print("  Подбор гиперпараметров (это займёт ~1-3 мин)...")

        res = fit_and_score(
            train_df, test_df,
            cfg["features"], tgt["col"], tgt["log"],
            tune=True,
        )

        print(f"\n  CV (train SC23+SC24):")
        for name in res["fitted"]:
            _print_metrics(name, res["cv_metrics"][name])

        print(f"\n  Test (SC25):")
        for name in res["fitted"]:
            _print_metrics(name, res["test_metrics"][name])

        # Наивный baseline
        y_tr = res["y_tr_raw"]
        y_te = res["test_true"]
        naive = 10 ** np.mean(np.log10(np.clip(y_tr, 1e-6, None))) if tgt["log"] else np.mean(y_tr)
        naive_m = compute_metrics(y_te, np.full_like(y_te, naive), tgt["log"])
        print(f"\n  Naive baseline: ", end="")
        _print_metrics("", naive_m)

        # Важности признаков
        print("\n  Расчёт важности признаков...")
        importances = compute_importances(res, cfg["features"])

        plots_sub = PLOTS_DIR / cfg["subdir"] / ("jmax" if tgt["log"] else "tdelta")
        plot_importances(importances, cfg["f_labels"],
                         title=f"Важность признаков -- {cfg['label']} -- {tgt['label']}",
                         out_path=plots_sub / "importance.png")
        plot_scatter(res, cfg["f_labels"],
                     title=f"{tgt['label']} -- прогноз vs факт (SC25)",
                     out_path=plots_sub / "scatter.png",
                     log_scale=tgt["log"])
        plot_residuals(res,
                       title=f"{tgt['label']}",
                       out_path=plots_sub / "residuals.png",
                       log_target=tgt["log"])

        results[tgt["col"]] = res

    return results


def print_comparison(all_results):
    """Сводная таблица: baseline vs best для каждого таргета."""
    print(f"\n{'='*65}")
    print("СВОДНАЯ ТАБЛИЦА: baseline vs best (после тюнинга)")
    print("="*65)

    for tgt in TARGETS:
        primary = "RMSLE_log10" if tgt["log"] else "RMSE"
        print(f"\n{tgt['label']}  [{primary}]")
        print(f"  {'Модель':<14} {'baseline':>12}  {'best':>12}  {'delta':>10}")
        print("  " + "-"*52)

        models = list(all_results["baseline"][tgt["col"]]["fitted"].keys())
        for name in models:
            m_base = all_results["baseline"][tgt["col"]]["test_metrics"][name]
            m_best = all_results["best"][tgt["col"]]["test_metrics"][name]
            v_base = m_base.get(primary, m_base.get("RMSE"))
            v_best = m_best.get(primary, m_best.get("RMSE"))
            delta  = v_best - v_base
            sign   = "+" if delta > 0 else ""
            print(f"  {name:<14} {v_base:>12.4f}  {v_best:>12.4f}  {sign}{delta:>8.4f}")


def main():
    warnings.filterwarnings("ignore")

    print(f"Загрузка: {INPUT_XLSX!r}")
    df = build_features(pd.read_excel(INPUT_XLSX, sheet_name=SHEET))

    cycle    = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    train_df = df[cycle.isin([23, 24])].copy()
    test_df  = df[cycle.isin([25])].copy()
    train_df = train_df[train_df["Jmax"].fillna(0) >= 10].copy()
    test_df  = test_df[test_df["Jmax"].fillna(0) >= 10].copy()
    print(f"Train SC23+SC24: {len(train_df)}  |  Test SC25: {len(test_df)}  (Jmax>=10)")

    all_results = {}
    for cfg_name, cfg in CONFIGS.items():
        all_results[cfg_name] = run_config(cfg_name, cfg, train_df, test_df)

    print_comparison(all_results)
    print(f"\nГрафики сохранены в: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
