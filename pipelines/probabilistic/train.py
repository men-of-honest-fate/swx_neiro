"""
Вероятностное обучение: одна конфигурация признаков, два целевых признака.
Выводит таблицу метрик (coverage, width, winkler, crps) и строит графики.

Запуск:  python train.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import (
    load_data, make_prob_models, prob_fit_and_score, COVERAGE,
    plot_intervals, plot_calibration, plot_metrics_bar,
)

warnings.filterwarnings("ignore")

# ── Конфигурация ──────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent.parent
PIPELINE     = Path(__file__).parent
FEATURE_COLS = ["helio_lon", "log_flare_power", "log_cme_velocity"]
PLOTS_DIR    = PIPELINE / "plots" / "train"

TARGETS = [
    dict(col="Jmax",    log=True,  unit="log10(pfu)", label="Jmax"),
    dict(col="T_delta", log=False, unit="часы",        label="T_delta"),
]

# ── Вспомогательные функции ───────────────────────────────────────────────────

METRIC_HEADERS = ["coverage", "width", "winkler", "pinball", "crps"]
HEADER_LABELS  = {
    "coverage": "Coverage",
    "width":    "Width",
    "winkler":  "Winkler↓",
    "pinball":  "Pinball↓",
    "crps":     "CRPS↓",
}


def _fmt(v, key):
    if np.isnan(v):
        return "    N/A"
    if key == "coverage":
        return f"{v:>7.0%}"
    return f"{v:>7.3f}"


def print_metrics_table(result: dict, target_label: str, unit: str, phase: str):
    key = f"{phase}_metrics"
    note = f"{'CV (cross-cycle)' if result['cv_mode']=='cross_cycle' else 'CV (StratKFold)'}"
    label = note if phase == "cv" else "Test SC25"
    header_line = f"{'Модель':<18}" + "".join(f"{HEADER_LABELS[k]:>8}" for k in METRIC_HEADERS)
    print(f"\n  {label}:")
    print("  " + header_line)
    print("  " + "-" * len(header_line))
    for name, m in result[key].items():
        row = f"  {name:<18}"
        for k in METRIC_HEADERS:
            row += _fmt(m.get(k, np.nan), k)
        print(row)
    if phase == "cv" and result.get("cycle_names"):
        print(f"  [фолды: {', '.join(result['cycle_names'])} → leave-one-cycle-out]")
    print(f"  [метрики в пространстве: {unit}]")


def save_results(all_results: dict, out_path: Path):
    rows = []
    for tgt_label, (result, unit) in all_results.items():
        for phase in ("cv", "test"):
            key = f"{phase}_metrics"
            for model, m in result[key].items():
                rows.append({
                    "target": tgt_label, "phase": phase, "model": model, **m
                })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"\nРезультаты: '{out_path}'")


# ── Главная функция ───────────────────────────────────────────────────────────

def main():
    train_df, test_df = load_data()
    print(f"\nПризнаки: {FEATURE_COLS}")
    print(f"Целевое покрытие: {COVERAGE:.0%}")

    models     = make_prob_models()
    all_results = {}

    for tgt in TARGETS:
        col   = tgt["col"]
        log   = tgt["log"]
        unit  = tgt["unit"]
        label = tgt["label"]

        print(f"\n{'='*55}")
        print(f"  Цель: {label}  ({unit})")
        print(f"{'='*55}")

        result = prob_fit_and_score(
            train_df, test_df,
            feature_cols=FEATURE_COLS,
            target_col=col,
            log_target=log,
            models={k: v for k, v in models.items()},  # свежие копии не нужны — clone внутри
        )

        print_metrics_table(result, label, unit, "cv")
        if result["has_test"]:
            print_metrics_table(result, label, unit, "test")

        sub = PLOTS_DIR / label.lower()

        # 1. Интервальный график (test)
        plot_intervals(
            result,
            title=f"80% интервалы прогноза — {label} (Test SC25)",
            out_path=sub / "intervals_test.png",
            ylabel=unit,
        )

        # 2. CV интервалы — подменяем test на cv данные для visualisation
        cv_result_for_plot = dict(result)
        cv_result_for_plot["test_intervals"]    = result["cv_intervals"]
        cv_result_for_plot["test_true_metric"]  = result["y_tr_metric"]
        cv_result_for_plot["test_metrics"]      = result["cv_metrics"]
        cv_result_for_plot["has_test"]          = True
        plot_intervals(
            cv_result_for_plot,
            title=f"80% интервалы прогноза — {label} (CV)",
            out_path=sub / "intervals_cv.png",
            ylabel=unit,
        )

        # 3. Диаграмма калибровки (только Gaussian-модели)
        plot_calibration(
            result,
            title=f"Калибровка {label} (CV)",
            out_path=sub / "calibration.png",
        )

        all_results[label] = (result, unit)

    # Итоговая таблица в Excel
    save_results(all_results, PIPELINE / "results" / "prob_results.xlsx")


if __name__ == "__main__":
    main()
