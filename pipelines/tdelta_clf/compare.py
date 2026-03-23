"""
Сравнение наборов признаков для классификации T_delta (Быстрые / Умеренные / Медленные).
Первичная метрика: Log Loss (меньше = лучше).

Запуск:  python compare.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    load_data, make_clf_models, clf_fit_and_score,
    CLASS_SHORT, MODEL_COLORS, N_CLASSES,
)

warnings.filterwarnings("ignore")

PIPELINE = Path(__file__).parent

FEATURE_SETS = [
    ("Базовая",           ["helio_lon", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Обе координаты",    ["helio_lon", "helio_lat", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Без КВМ",           ["helio_lon", "log_goes_peak_flux"]),
    ("T_delta_flare",     ["t_delta_flare", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Все базовые",       ["helio_lon", "helio_lat", "t_delta_flare",
                           "log_goes_peak_flux", "log_cme_velocity"]),
    ("Флюэс вместо пика", ["helio_lon", "log_fluence", "log_cme_velocity"]),
    ("Флюэс + пик",       ["helio_lon", "log_fluence", "log_goes_peak_flux", "log_cme_velocity"]),
    ("КВМ расширенный",   ["helio_lon", "log_goes_peak_flux", "log_cme_velocity",
                           "cme_width_deg", "cme_pa_deg"]),
    ("Вспышка расшир.",   ["helio_lon", "log_fluence", "log_goes_peak_flux",
                           "log_flare_dur_min", "log_cme_velocity"]),
    ("Kitchen sink",      ["helio_lon", "helio_lat", "t_delta_flare",
                           "log_fluence", "log_goes_peak_flux", "log_flare_dur_min",
                           "log_cme_velocity", "cme_width_deg", "cme_pa_deg"]),
    ("Пик+нараст.",       ["helio_lon", "log_goes_peak_flux",
                           "log_goes_rise_min", "log_cme_velocity"]),
    ("GOES полный",       ["helio_lon", "helio_lat", "log_goes_peak_flux",
                           "log_fluence", "log_cme_velocity"]),
    ("GOES KS",           ["helio_lon", "helio_lat", "log_goes_peak_flux",
                           "log_fluence", "log_flare_dur_min",
                           "log_cme_velocity", "cme_width_deg", "cme_pa_deg", "t_delta_flare"]),
]

MODEL_ORDER = ["LogReg", "Forest", "ExtraTrees", "Boosting", "SVC"]
PLOTS_DIR   = PIPELINE / "plots" / "compare"
PRIMARY     = "log_loss"


def run_all(train_df, test_df):
    models = make_clf_models()
    rows   = []

    for fs_label, feat_cols in FEATURE_SETS:
        print(f"  {fs_label:<30} ...", end=" ", flush=True)
        try:
            result = clf_fit_and_score(train_df, test_df, feat_cols, models=models)
        except Exception as e:
            print(f"ОШИБКА: {e}")
            continue

        for model in MODEL_ORDER:
            cv_m   = result["cv_metrics"].get(model, {})
            test_m = result["test_metrics"].get(model, {})
            rows.append(dict(
                feature_set=fs_label, model=model,
                cv_accuracy=cv_m.get("accuracy", np.nan),
                cv_log_loss=cv_m.get("log_loss", np.nan),
                cv_brier=cv_m.get("brier", np.nan),
                cv_auc=cv_m.get("auc_macro", np.nan),
                test_accuracy=test_m.get("accuracy", np.nan),
                test_log_loss=test_m.get("log_loss", np.nan),
                test_brier=test_m.get("brier", np.nan),
                test_auc=test_m.get("auc_macro", np.nan),
            ))
        print("OK")

    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, phase: str, metric: str, out_path: Path):
    col = f"{phase}_{metric}"
    pivot = df.pivot_table(index="feature_set", columns="model",
                           values=col, aggfunc="min")
    pivot = pivot.reindex(columns=[m for m in MODEL_ORDER if m in pivot.columns])
    pivot = pivot.reindex([fs for fs, _ in FEATURE_SETS if fs in pivot.index])

    cmap = "YlOrRd" if metric in ("log_loss", "brier") else "YlOrRd_r"

    fig, ax = plt.subplots(figsize=(len(MODEL_ORDER) * 1.5 + 1.5, len(pivot) * 0.5 + 1.5))
    im = ax.imshow(pivot.values.astype(float), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    vmin = np.nanmin(pivot.values)
    vmax = np.nanmax(pivot.values)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            txt = "N/A" if np.isnan(v) else f"{v:.3f}"
            rel = (v - vmin) / (vmax - vmin + 1e-9) if not np.isnan(v) else 0.5
            clr = "white" if rel > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=clr)

    plt.colorbar(im, ax=ax, pad=0.02)
    phase_lbl = "CV" if phase == "cv" else "Test SC25"
    ax.set_title(f"T_delta классификация — {metric} ({phase_lbl})", fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_best_per_featureset(df: pd.DataFrame, phase: str, out_path: Path):
    col = f"{phase}_{PRIMARY}"
    fs_labels, best_vals, best_models = [], [], []

    for fs_label, _ in FEATURE_SETS:
        sub = df[df["feature_set"] == fs_label].dropna(subset=[col])
        if sub.empty:
            fs_labels.append(fs_label); best_vals.append(np.nan); best_models.append("N/A")
            continue
        idx = sub[col].idxmin()
        fs_labels.append(fs_label)
        best_vals.append(sub.loc[idx, col])
        best_models.append(sub.loc[idx, "model"])

    x      = np.arange(len(fs_labels))
    colors = [MODEL_COLORS.get(m, "#ccc") for m in best_models]
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(x, best_vals, color=colors, edgecolor="white", width=0.65)

    for bar, v, m in zip(bars, best_vals, best_models):
        if np.isnan(v): continue
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(best_vals or [1]) * 0.01,
                f"{m}\n{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(fs_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Log Loss (лучшая модель)")
    phase_lbl = "CV" if phase == "cv" else "Test SC25"
    ax.set_title(f"T_delta классификация — лучший Log Loss по набору признаков ({phase_lbl})")
    ax.grid(axis="y", alpha=0.3)

    seen = {}
    for m, c in zip(best_models, colors):
        if m not in seen: seen[m] = c
    handles = [Patch(facecolor=c, label=m) for m, c in seen.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def print_summary(df: pd.DataFrame, phase: str):
    col     = f"{phase}_{PRIMARY}"
    acc_col = f"{phase}_accuracy"
    auc_col = f"{phase}_auc"
    phase_lbl = "CV" if phase == "cv" else "Test SC25"
    print(f"\nT_delta Log Loss ({phase_lbl}), меньше = лучше:")

    rows = []
    for fs_label, _ in FEATURE_SETS:
        sub = df[df["feature_set"] == fs_label].dropna(subset=[col])
        if sub.empty:
            rows.append((fs_label, np.nan, np.nan, np.nan, "N/A"))
            continue
        idx = sub[col].idxmin()
        r   = sub.loc[idx]
        rows.append((fs_label, r[col], r[acc_col], r[auc_col], r["model"]))

    rows.sort(key=lambda r: r[1] if not np.isnan(r[1]) else 9999)
    for fs, ll, acc, auc, m in rows:
        if np.isnan(ll):
            print(f"  {fs:<35}  N/A")
        else:
            print(f"  {fs:<35}  LogLoss={ll:.3f}  Acc={acc:.0%}  AUC={auc:.3f}  [{m}]")


def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    train_df, test_df = load_data()
    print("\nОбучение классификаторов (T_delta) по всем конфигурациям...")
    df = run_all(train_df, test_df)

    print("\nПостроение графиков...")
    for phase in ("cv", "test"):
        for metric in ("log_loss", "accuracy", "auc"):
            plot_heatmap(df, phase, metric,
                         PLOTS_DIR / f"heatmap_{metric}_{phase}.png")
        plot_best_per_featureset(df, phase, PLOTS_DIR / f"best_{phase}.png")

    out_xlsx = PIPELINE / "results" / "comparison_results.xlsx"
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)
    print(f"\nТаблица: '{out_xlsx}'")
    print(f"Графики: {PLOTS_DIR}/")

    print("\n" + "=" * 60)
    print_summary(df, "cv")
    print_summary(df, "test")


if __name__ == "__main__":
    main()
