"""
Классификация СПС: одна конфигурация признаков, подробный вывод.

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
    load_data, make_clf_models, clf_fit_and_score,
    CLASS_SHORT, CLASS_LABELS, N_CLASSES, BINS,
    plot_confusion_matrices, plot_roc_curves,
    plot_prob_heatmap, plot_calibration_clf,
)

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent.parent.parent
PIPELINE     = Path(__file__).parent
FEATURE_COLS = ["helio_lon", "log_flare_power", "log_cme_velocity"]
PLOTS_DIR    = PIPELINE / "plots" / "train"

METRIC_HEADERS = ["accuracy", "log_loss", "brier", "auc_macro"]
HEADER_LABELS  = {
    "accuracy":  "Accuracy",
    "log_loss":  "LogLoss↓",
    "brier":     "Brier↓",
    "auc_macro": "AUC↑",
}


def print_metrics_table(result: dict, phase: str):
    key  = f"{phase}_metrics"
    label = (f"CV ({result['cv_mode'].replace('_',' ')})"
             if phase == "cv" else "Test SC25")
    header = f"  {'Модель':<18}" + "".join(f"{HEADER_LABELS[k]:>9}" for k in METRIC_HEADERS)
    print(f"\n  {label}:")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in result[key].items():
        row = f"  {name:<18}"
        for k in METRIC_HEADERS:
            v = m.get(k, np.nan)
            row += f"{'N/A':>9}" if np.isnan(v) else f"{v:>9.3f}"
        print(row)
    if phase == "cv" and result.get("cycle_names"):
        print(f"  [leave-one-cycle-out: {', '.join(result['cycle_names'])}]")


def print_prob_table(result: dict):
    """Таблица вероятностей для тестовых событий."""
    if not result["has_test"] or len(result["test_true"]) == 0:
        return

    y_true  = result["test_true"]
    jmax    = result["test_jmax"]
    dates   = result["test_dates"]
    probs   = result["test_probs"]
    models  = list(probs.keys())

    sort_idx = np.argsort(jmax)
    print(f"\n  Прогнозы на Test SC25 (отсортировано по Jmax):")
    print(f"  {'Дата':<12} {'Jmax':>7} {'Факт':<8}", end="")
    for name in models:
        print(f"  {name:>12}  [P(S1) P(S3) P(S4)]", end="")
    print()
    print("  " + "-" * (12 + 7 + 8 + len(models) * 32))

    for i in sort_idx:
        date_s = str(dates[i])[:10] if len(dates) > 0 else "?"
        true_lbl = CLASS_SHORT[y_true[i]]
        row = f"  {date_s:<12} {jmax[i]:>7.0f} {true_lbl:<8}"
        for name in models:
            p = probs[name][i]
            pred = CLASS_SHORT[np.argmax(p)]
            ok = "✓" if np.argmax(p) == y_true[i] else "✗"
            row += f"  {ok}{pred:>6}  [{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}]"
        print(row)


def save_results(result: dict, out_path: Path):
    rows = []
    for phase in ("cv", "test"):
        y_true = result["cv_true"] if phase == "cv" else result["test_true"]
        probs  = result["cv_probs"] if phase == "cv" else result["test_probs"]
        if len(y_true) == 0:
            continue
        for name, p in probs.items():
            for i, (yt, pi) in enumerate(zip(y_true, p)):
                rows.append(dict(
                    phase=phase, model=name, idx=i,
                    true_class=CLASS_SHORT[int(yt)],
                    pred_class=CLASS_SHORT[int(np.argmax(pi))],
                    correct=int(np.argmax(pi)) == int(yt),
                    **{f"P_{CLASS_SHORT[k].replace('-','_')}": pi[k]
                       for k in range(N_CLASSES)},
                ))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(out_path, index=False)
    print(f"\nРезультаты: '{out_path}'")


def main():
    train_df, test_df = load_data()

    print(f"\nПризнаки: {FEATURE_COLS}")
    print(f"Классы: {' | '.join(CLASS_SHORT)} (границы: {BINS[:-1]} pfu)")

    models = make_clf_models()
    result = clf_fit_and_score(train_df, test_df,
                               feature_cols=FEATURE_COLS,
                               models=models)

    print("\n" + "=" * 55)
    print_metrics_table(result, "cv")
    if result["has_test"]:
        print_metrics_table(result, "test")

    print_prob_table(result)

    # Графики
    print("\nПостроение графиков...")
    plot_confusion_matrices(
        result, "Матрица ошибок — Test SC25",
        PLOTS_DIR / "confusion_test.png", phase="test")
    plot_confusion_matrices(
        result, "Матрица ошибок — CV",
        PLOTS_DIR / "confusion_cv.png", phase="cv")
    plot_roc_curves(
        result, "ROC-кривые — Test SC25",
        PLOTS_DIR / "roc_test.png", phase="test")
    plot_roc_curves(
        result, "ROC-кривые — CV",
        PLOTS_DIR / "roc_cv.png", phase="cv")
    plot_prob_heatmap(
        result, "Вероятности прогноза — Test SC25",
        PLOTS_DIR / "prob_heatmap.png")
    plot_calibration_clf(
        result, "Калибровка (CV)",
        PLOTS_DIR / "calibration_cv.png")

    save_results(result, PIPELINE / "results" / "clf_results.xlsx")
    (PIPELINE / "results").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
