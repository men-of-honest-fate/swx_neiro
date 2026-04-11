"""
Вероятностное сравнение наборов признаков.
Та же идеология, что и compare.py, но метрика — Winkler score (интервальный прогноз).

Запуск:  python compare.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils import (
    load_data, make_prob_models, prob_fit_and_score, COVERAGE,
    MODEL_COLORS,
)

warnings.filterwarnings("ignore")

# ── Наборы признаков (те же 10, что в compare.py) ────────────────────────────

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

TARGETS = [
    dict(col="Jmax",    log=True,  unit="log10(pfu)", label="Jmax"),
    dict(col="T_delta", log=False, unit="часы",        label="T_delta"),
]

ROOT        = Path(__file__).parent.parent.parent
PIPELINE    = Path(__file__).parent
MODEL_ORDER = ["QuantLinear", "QuantBoosting", "BayesRidge", "GPR_RBF", "ConformalRF"]
PLOTS_DIR   = PIPELINE / "plots" / "compare"
PRIMARY     = "winkler"  # главная метрика для сравнения


# ── Обучение ──────────────────────────────────────────────────────────────────

def run_all(train_df, test_df):
    models = make_prob_models()
    rows   = []

    for tgt in TARGETS:
        col, log, unit, label = tgt["col"], tgt["log"], tgt["unit"], tgt["label"]
        print(f"\n[{label}]")

        for fs_label, feat_cols in FEATURE_SETS:
            print(f"  {fs_label:<30} ...", end=" ", flush=True)
            try:
                result = prob_fit_and_score(
                    train_df, test_df, feat_cols, col, log,
                    models={k: v for k, v in models.items()},
                )
            except Exception as e:
                print(f"ОШИБКА: {e}")
                continue

            for model in MODEL_ORDER:
                cv_m   = result["cv_metrics"].get(model, {})
                test_m = result["test_metrics"].get(model, {})
                rows.append(dict(
                    target=label, feature_set=fs_label, model=model,
                    cv_coverage=cv_m.get("coverage", np.nan),
                    cv_width=cv_m.get("width", np.nan),
                    cv_winkler=cv_m.get("winkler", np.nan),
                    cv_pinball=cv_m.get("pinball", np.nan),
                    cv_crps=cv_m.get("crps", np.nan),
                    test_coverage=test_m.get("coverage", np.nan),
                    test_width=test_m.get("width", np.nan),
                    test_winkler=test_m.get("winkler", np.nan),
                    test_pinball=test_m.get("pinball", np.nan),
                    test_crps=test_m.get("crps", np.nan),
                ))
            print("OK")

    return pd.DataFrame(rows)


# ── Графики ───────────────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, target: str, phase: str, metric: str,
                 out_path: Path, cmap="YlOrRd_r"):
    sub = df[(df["target"] == target)].copy()
    col = f"{phase}_{metric}"
    pivot = sub.pivot_table(index="feature_set", columns="model",
                            values=col, aggfunc="min")
    pivot = pivot.reindex(columns=[m for m in MODEL_ORDER if m in pivot.columns])
    pivot = pivot.reindex([fs for fs, _ in FEATURE_SETS if fs in pivot.index])

    fig, ax = plt.subplots(figsize=(len(MODEL_ORDER) * 1.6 + 1.5, len(pivot) * 0.55 + 1.5))
    im = ax.imshow(pivot.values.astype(float), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    vmin = np.nanmin(pivot.values)
    vmax = np.nanmax(pivot.values)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if np.isnan(v):
                txt, clr = "N/A", "gray"
            else:
                txt = f"{v:.3f}"
                clr = "white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=clr)

    plt.colorbar(im, ax=ax, pad=0.02)
    phase_label = "CV" if phase == "cv" else "Test SC25"
    ax.set_title(f"{target} — {metric.capitalize()} ({phase_label})", fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_coverage_vs_width(df: pd.DataFrame, target: str, phase: str, out_path: Path):
    """Coverage vs Width: хорошая модель — высокое покрытие при малой ширине."""
    sub = df[df["target"] == target].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvline(COVERAGE, color="gray", ls="--", lw=1, alpha=0.6, label=f"{COVERAGE:.0%} цель")

    for model in MODEL_ORDER:
        ms = sub[sub["model"] == model]
        cov = ms[f"{phase}_coverage"].values
        wid = ms[f"{phase}_width"].values
        mask = ~(np.isnan(cov) | np.isnan(wid))
        if mask.sum() == 0:
            continue
        ax.scatter(cov[mask], wid[mask],
                   s=60, color=MODEL_COLORS.get(model, "#333"),
                   label=model, alpha=0.8, edgecolors="w", lw=0.5, zorder=4)

    ax.set_xlabel("Coverage (PICP)")
    ax.set_ylabel("Width (ширина интервала)")
    ax.set_title(f"{target} — Coverage vs Width ({'CV' if phase=='cv' else 'Test SC25'})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_best_per_featureset(df: pd.DataFrame, target: str, phase: str, out_path: Path):
    """Лучшая модель (мин. Winkler) для каждого набора признаков."""
    sub = df[df["target"] == target].copy()
    col = f"{phase}_winkler"

    fs_labels, best_winklers, best_models = [], [], []
    for fs_label, _ in FEATURE_SETS:
        fs_sub = sub[(sub["feature_set"] == fs_label)].dropna(subset=[col])
        if fs_sub.empty:
            fs_labels.append(fs_label)
            best_winklers.append(np.nan)
            best_models.append("N/A")
            continue
        idx = fs_sub[col].idxmin()
        fs_labels.append(fs_label)
        best_winklers.append(fs_sub.loc[idx, col])
        best_models.append(fs_sub.loc[idx, "model"])

    x = np.arange(len(fs_labels))
    colors = [MODEL_COLORS.get(m, "#ccc") for m in best_models]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(x, best_winklers, color=colors, edgecolor="white", width=0.65)
    for bar, v, m in zip(bars, best_winklers, best_models):
        if np.isnan(v):
            continue
        ax.text(bar.get_x() + bar.get_width()/2, v + max(best_winklers or [1]) * 0.01,
                f"{m}\n{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(fs_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Winkler score (лучшая модель)")
    ax.set_title(f"{target} — лучший Winkler по набору признаков "
                 f"({'CV' if phase=='cv' else 'Test SC25'})")
    ax.grid(axis="y", alpha=0.3)

    # Легенда моделей
    from matplotlib.patches import Patch
    seen = {}
    for m, c in zip(best_models, colors):
        if m not in seen:
            seen[m] = c
    handles = [Patch(facecolor=c, label=m) for m, c in seen.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Вывод сводки ──────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, target: str, phase: str):
    sub = df[df["target"] == target]
    col = f"{phase}_winkler"
    cov_col = f"{phase}_coverage"
    print(f"\n{target} ({col}, меньше = лучше):")

    rows = []
    for fs_label, _ in FEATURE_SETS:
        fs_sub = sub[sub["feature_set"] == fs_label].dropna(subset=[col])
        if fs_sub.empty:
            rows.append((fs_label, np.nan, np.nan, "N/A"))
            continue
        idx      = fs_sub[col].idxmin()
        best_row = fs_sub.loc[idx]
        rows.append((fs_label, best_row[col], best_row[cov_col], best_row["model"]))

    rows.sort(key=lambda r: r[1] if not np.isnan(r[1]) else 9999)
    for fs, wink, cov, model in rows:
        if np.isnan(wink):
            print(f"  {fs:<35} N/A (нет данных SC25)")
        else:
            print(f"  {fs:<35} Winkler={wink:.3f}  Coverage={cov:.0%}  [{model}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    train_df, test_df = load_data()
    print(f"\nЦелевое покрытие: {COVERAGE:.0%}")
    print("\nОбучение вероятностных моделей по всем конфигурациям...")
    df = run_all(train_df, test_df)

    print("\nПостроение графиков...")
    for tgt in TARGETS:
        label = tgt["label"]
        for phase in ("cv", "test"):
            plot_heatmap(df, label, phase, PRIMARY,
                         PLOTS_DIR / f"heatmap_{label.lower()}_{phase}.png")
            plot_coverage_vs_width(df, label, phase,
                                   PLOTS_DIR / f"cov_width_{label.lower()}_{phase}.png")
            plot_best_per_featureset(df, label, phase,
                                     PLOTS_DIR / f"best_{label.lower()}_{phase}.png")

    out_xlsx = PIPELINE / "results" / "comparison_results.xlsx"
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)
    print(f"\nТаблица: '{out_xlsx}'")
    print(f"Графики: {PLOTS_DIR}/")

    print("\n" + "="*60)
    print(" Сводка результатов")
    print("="*60)
    for tgt in TARGETS:
        label = tgt["label"]
        print_summary(df, label, "cv")
        if tgt["col"] == "Jmax":  # test только для таргетов с SC25
            print_summary(df, label, "test")


if __name__ == "__main__":
    main()
