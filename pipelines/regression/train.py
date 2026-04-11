"""
Обучение моделей СПС — базовая конфигурация признаков.

Входные признаки (log10-трансформация для широкодиапазонных):
    helio_lon           -- гелиодолгота (W > 0, E < 0)
    log_flare_power     -- log10(интенсивность вспышки, Вт/м²)
    log_cme_velocity    -- log10(скорость КВМ, км/с)

Целевые переменные:
    Jmax     -- оценивается в log10-пространстве (RMSLE_log10, R2_log, Spearman)
    T_delta  -- RMSE, R2, Spearman

Train: SC23 + SC24  |  Test: SC25

Флаги:
    TUNE = True   -- включить RandomizedSearchCV для подбора гиперпараметров
                     (занимает ~2–3 мин дополнительно)

Запуск: python train.py
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
    COL_CYCLE, COL_COORDS, COL_CLASS, COL_CME, COL_JMAX, COL_TDELTA,
)

ROOT        = Path(__file__).parent.parent.parent
PIPELINE    = Path(__file__).parent
INPUT_XLSX  = ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx"
SHEET       = "Флюэс GOES"
OUTPUT_XLSX = PIPELINE / "results" / "predictions.xlsx"
PLOTS_DIR   = PIPELINE / "plots"

TUNE = False   # True = включить подбор гиперпараметров

# log10-трансформированные признаки: выравнивают диапазоны для линейных моделей
# и повышают R² для всего ансамбля (RF/GBR и так инвариантны, Linear/Ridge выигрывают)
FEATURE_COLS   = ["helio_lon", "log_flare_power", "log_cme_velocity"]
FEATURE_LABELS = ["Гелиодолгота", "log10(интенс. вспышки)", "log10(скор. КВМ)"]


def _print_metrics(name: str, m: dict):
    parts = [f"RMSE={m['RMSE']:.2f}"]
    if "RMSLE_log10" in m:
        parts.append(f"RMSLE_log10={m['RMSLE_log10']:.3f}")
    if "R2_log" in m:
        parts.append(f"R2_log={m['R2_log']:.3f}")
    if "R2" in m:
        parts.append(f"R2={m['R2']:.3f}")
    if "Spearman" in m:
        parts.append(f"Spearman={m['Spearman']:.3f}")
    if "MedAE" in m:
        parts.append(f"MedAE={m['MedAE']:.2f}")
    print(f"  {name:<12} " + "  ".join(parts))


def run_target(train_df, test_df, target_col, target_label, log_target, subdir):
    print(f"\n{'='*60}")
    print(f"Цель: {target_label}  (log_target={log_target})")
    print("="*60)

    result = fit_and_score(
        train_df, test_df, FEATURE_COLS, target_col, log_target, tune=TUNE
    )

    # ── Per-fold cross-cycle CV ──────────────────────────────────────────────
    cycle_names = result.get("cycle_names", [])
    fold_metrics = result.get("fold_metrics", {})
    primary = "RMSLE_log10" if log_target else "RMSE"

    if cycle_names and fold_metrics:
        print(f"\n--- Cross-Cycle CV (leave-one-cycle-out) ---")
        header = f"  {'Модель':<12}" + "".join(f"  val={c:>4}" for c in cycle_names) + "   mean    std"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, folds in fold_metrics.items():
            vals = [f.get(primary, f.get("RMSE")) for f in folds]
            fold_str = "".join(f"  {v:>8.4f}" for v in vals)
            print(f"  {name:<12}{fold_str}  {np.mean(vals):>7.4f}  {np.std(vals):>5.4f}")

    print(f"\n--- CV aggregate ({result['cv_mode']}) ---")
    for name in result["fitted"]:
        _print_metrics(name, result["cv_metrics"][name])

    print(f"\n--- Test (SC25) ---")
    for name in result["fitted"]:
        _print_metrics(name, result["test_metrics"][name])

    # Наивный базовый уровень: предсказание среднего тренировочного значения
    y_tr = result["y_tr_raw"]
    y_te = result["test_true"]
    if log_target:
        naive = 10 ** np.mean(np.log10(np.clip(y_tr, 1e-6, None)))
    else:
        naive = np.mean(y_tr)
    from spe_utils import compute_metrics
    naive_m = compute_metrics(y_te, np.full_like(y_te, naive), log_target)
    print(f"\n  Naive baseline (mean train): ", end="")
    _print_metrics("", naive_m)

    print("\n  Расчёт важности признаков...")
    importances = compute_importances(result, FEATURE_COLS)

    plots_sub = PLOTS_DIR / subdir
    plot_importances(importances, FEATURE_LABELS,
                     title=f"Вклад признаков -- {target_label}",
                     out_path=plots_sub / "importance.png")
    plot_scatter(result, FEATURE_LABELS,
                 title=f"{target_label} -- прогноз vs факт (SC25)",
                 out_path=plots_sub / "scatter.png",
                 log_scale=log_target)
    plot_residuals(result,
                   title=target_label,
                   out_path=plots_sub / "residuals.png",
                   log_target=log_target)

    return result


def main():
    warnings.filterwarnings("ignore")

    print(f"Загрузка: {INPUT_XLSX!r}")
    df = build_features(
        pd.read_excel(INPUT_XLSX, sheet_name=SHEET)
    )

    cycle = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    train_df = df[cycle.isin([23, 24])].copy()
    test_df  = df[cycle.isin([25])].copy()
    train_df = train_df[train_df["Jmax"].fillna(0) >= 10].copy()
    test_df  = test_df[test_df["Jmax"].fillna(0) >= 10].copy()
    print(f"Train SC23+SC24: {len(train_df)}  |  Test SC25: {len(test_df)}  (Jmax>=10)")
    print(f"Hyperparameter tuning: {'ON' if TUNE else 'OFF (set TUNE=True to enable)'}")

    res_j = run_target(train_df, test_df,
                       "Jmax", "Jmax (pfu)",
                       log_target=True, subdir="jmax")

    res_t = run_target(train_df, test_df,
                       "T_delta", "T_delta (часы)",
                       log_target=False, subdir="tdelta")

    # Сохранение предсказаний
    base_cols = [c for c in [COL_CYCLE, "Дата события", COL_COORDS,
                              COL_CLASS, COL_CME, COL_JMAX, COL_TDELTA]
                 if c in df.columns]
    out = test_df[base_cols].reset_index(drop=True)

    def _fill(result, suffix, test_df_ref):
        mapping = {orig_i: pos for pos, orig_i in enumerate(test_df_ref.index)}
        n = len(out)
        for name, pred in result["test_preds"].items():
            col = f"{name}_{suffix}"
            series = pd.Series(np.nan, index=range(n))
            for orig_i, val in zip(result["test_idx"], pred):
                if orig_i in mapping:
                    series.iloc[mapping[orig_i]] = val
            out[col] = series.values

    _fill(res_j, "Jmax",   test_df)
    _fill(res_t, "Tdelta", test_df)

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(OUTPUT_XLSX, index=False)
    print(f"\nПредсказания: {OUTPUT_XLSX!r}")
    print(f"Графики:      {PLOTS_DIR}/jmax/  и  {PLOTS_DIR}/tdelta/")


if __name__ == "__main__":
    main()
