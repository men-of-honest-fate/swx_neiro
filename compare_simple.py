"""
Упрощённый сравнительный анализ: 4 модели × 4 набора признаков.

Модели:  Linear (LogReg для классификации), RandomForest, Boosting, SVR (SVC для классификации)
Наборы:
  1. Базовая             — helio_lon, log_goes_peak_flux, log_cme_velocity
  2. Флюэс вместо пика   — helio_lon, log_fluence, log_cme_velocity
  3. Обе координаты      — helio_lon, helio_lat, log_goes_peak_flux, log_cme_velocity
  4. Координаты + флюэс  — helio_lon, helio_lat, log_fluence, log_cme_velocity

Пайплайны: регрессия (Jmax + T_delta), классификация Jmax (S-шкала),
           классификация T_delta (Быстрые / Умеренные / Медленные).

Запуск:  python compare_simple.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spe_utils import build_features, fit_and_score, COL_CYCLE, MODEL_COLORS

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PLOTS_DIR = ROOT / "results_simple" / "plots"
OUT_XLSX  = ROOT / "results_simple" / "simple_results.xlsx"

# ── Конфигурация ──────────────────────────────────────────────────────────────

FEATURE_SETS = [
    ("Базовая",            ["helio_lon", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Флюэс вместо пика",  ["helio_lon", "log_fluence", "log_cme_velocity"]),
    ("Обе координаты",     ["helio_lon", "helio_lat", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Координаты+флюэс",   ["helio_lon", "helio_lat", "log_fluence", "log_cme_velocity"]),
]

REG_MODELS   = ["Linear", "Forest", "Boosting", "SVR"]
CLF_MODELS   = ["LogReg", "Forest", "Boosting", "SVC"]

# ── Загрузка данных ───────────────────────────────────────────────────────────

def load():
    df = build_features(
        pd.read_excel(ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
                      sheet_name="Флюэс GOES")
    )
    cycle = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    train = df[cycle.isin([23, 24]) & (df["Jmax"].fillna(0) >= 10)].copy()
    test  = df[cycle.isin([25]) & (df["Jmax"].fillna(0) >= 10)].copy()
    print(f"Train SC23+SC24: {len(train)}  |  Test SC25: {len(test)}")
    return train, test

# ── Регрессия ─────────────────────────────────────────────────────────────────

def run_regression(train_df, test_df):
    from spe_utils import fit_and_score
    rows = []
    for fs_label, fs_cols in FEATURE_SETS:
        for tgt_col, tgt_log in [("Jmax", True), ("T_delta", False)]:
            print(f"  [reg/{tgt_col}] {fs_label} ...", end=" ", flush=True)
            try:
                res = fit_and_score(train_df, test_df, fs_cols, tgt_col, tgt_log)
                for model in REG_MODELS:
                    cv_m  = res["cv_metrics"].get(model, {})
                    te_m  = res["test_metrics"].get(model, {})
                    primary_cv  = cv_m.get("RMSLE_log10", cv_m.get("RMSE", np.nan))  if tgt_log else cv_m.get("RMSE", np.nan)
                    primary_te  = te_m.get("RMSLE_log10", te_m.get("RMSE", np.nan))  if tgt_log else te_m.get("RMSE", np.nan)
                    rows.append(dict(
                        pipeline="regression", target=tgt_col,
                        feature_set=fs_label, model=model,
                        cv_primary=primary_cv,
                        test_primary=primary_te,
                        test_r2=te_m.get("R2_log", te_m.get("R2", np.nan)),
                        test_spearman=te_m.get("Spearman", np.nan),
                    ))
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
    return pd.DataFrame(rows)

# ── Классификация Jmax ────────────────────────────────────────────────────────

def run_jmax_clf(train_df, test_df):
    from pipelines.classification.utils import (
        make_clf_models, clf_fit_and_score, N_CLASSES,
    )

    # Оставляем только 4 модели
    all_models = make_clf_models()
    models = {k: v for k, v in all_models.items() if k in CLF_MODELS}

    import pandas as _pd
    import numpy as _np

    # Добавляем jmax_class
    from pipelines.classification.utils import jmax_to_class
    train_df = train_df.copy(); test_df = test_df.copy()
    train_df["jmax_class"] = jmax_to_class(train_df["Jmax"].values)
    test_df["jmax_class"]  = jmax_to_class(test_df["Jmax"].values)

    rows = []
    for fs_label, fs_cols in FEATURE_SETS:
        print(f"  [jmax_clf] {fs_label} ...", end=" ", flush=True)
        try:
            result = clf_fit_and_score(train_df, test_df, fs_cols, models=models)
            for model in CLF_MODELS:
                cv_m   = result["cv_metrics"].get(model, {})
                test_m = result["test_metrics"].get(model, {})
                rows.append(dict(
                    pipeline="jmax_clf", target="Jmax_class",
                    feature_set=fs_label, model=model,
                    cv_primary=cv_m.get("log_loss", _np.nan),
                    test_primary=test_m.get("log_loss", _np.nan),
                    test_accuracy=test_m.get("accuracy", _np.nan),
                    test_auc=test_m.get("auc_macro", _np.nan),
                ))
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
    return _pd.DataFrame(rows)

# ── Классификация T_delta ─────────────────────────────────────────────────────

def run_tdelta_clf(train_df, test_df):
    from pipelines.tdelta_clf.utils import (
        make_clf_models, clf_fit_and_score, tdelta_to_class,
    )
    import numpy as _np
    import pandas as _pd

    all_models = make_clf_models()
    models = {k: v for k, v in all_models.items() if k in CLF_MODELS}

    train_df = train_df.copy(); test_df = test_df.copy()
    train_df = train_df[train_df["T_delta"].notna()].copy()
    test_df  = test_df[test_df["T_delta"].notna()].copy()
    train_df["tdelta_class"] = tdelta_to_class(train_df["T_delta"].values)
    test_df["tdelta_class"]  = tdelta_to_class(test_df["T_delta"].values)

    rows = []
    for fs_label, fs_cols in FEATURE_SETS:
        print(f"  [tdelta_clf] {fs_label} ...", end=" ", flush=True)
        try:
            result = clf_fit_and_score(train_df, test_df, fs_cols, models=models)
            for model in CLF_MODELS:
                cv_m   = result["cv_metrics"].get(model, {})
                test_m = result["test_metrics"].get(model, {})
                rows.append(dict(
                    pipeline="tdelta_clf", target="T_delta_class",
                    feature_set=fs_label, model=model,
                    cv_primary=cv_m.get("log_loss", _np.nan),
                    test_primary=test_m.get("log_loss", _np.nan),
                    test_accuracy=test_m.get("accuracy", _np.nan),
                    test_auc=test_m.get("auc_macro", _np.nan),
                ))
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
    return _pd.DataFrame(rows)

# ── Визуализация ──────────────────────────────────────────────────────────────

_CLF_COLORS = {"LogReg": "#1f77b4", "Forest": "#8c564b",
               "Boosting": "#ff7f0e", "SVC": "#2ca02c"}
_ALL_COLORS = {**MODEL_COLORS, **_CLF_COLORS}

FS_LABELS = [fs for fs, _ in FEATURE_SETS]


def _bar_grouped(ax, sub, model_list, metric_col, ylabel, title):
    n_fs = len(FS_LABELS)
    n_m  = len(model_list)
    bar_w = 0.7 / n_m
    xg = np.arange(n_fs)

    for mi, model in enumerate(model_list):
        offset = (mi - (n_m - 1) / 2) * bar_w
        vals = []
        for fs in FS_LABELS:
            row = sub[(sub["feature_set"] == fs) & (sub["model"] == model)]
            vals.append(row[metric_col].values[0] if not row.empty else np.nan)
        ax.bar(xg + offset, vals, width=bar_w,
               color=_ALL_COLORS.get(model, "#888"),
               label=model, edgecolor="white", linewidth=0.5)

    ax.set_xticks(xg)
    ax.set_xticklabels(FS_LABELS, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)


def plot_regression(df_reg):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for tgt, log_tgt in [("Jmax", True), ("T_delta", False)]:
        sub = df_reg[df_reg["target"] == tgt]
        ylabel = "RMSLE log₁₀" if log_tgt else "RMSE (часы)"
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        _bar_grouped(axes[0], sub, REG_MODELS, "cv_primary",   ylabel, f"CV — {tgt}")
        _bar_grouped(axes[1], sub, REG_MODELS, "test_primary", ylabel, f"Test SC25 — {tgt}")
        fig.suptitle(f"Регрессия {tgt}: 4 модели × 4 набора признаков", fontsize=12)
        plt.tight_layout()
        path = PLOTS_DIR / f"reg_{tgt.lower()}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def plot_classification(df_clf, pipeline_name, target_label):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sub = df_clf[df_clf["pipeline"] == pipeline_name]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    _bar_grouped(axes[0, 0], sub, CLF_MODELS, "cv_primary",   "Log Loss", f"CV Log Loss")
    _bar_grouped(axes[0, 1], sub, CLF_MODELS, "test_primary", "Log Loss", f"Test Log Loss (SC25)")
    _bar_grouped(axes[1, 0], sub, CLF_MODELS, "test_accuracy","Accuracy", f"Test Accuracy")
    _bar_grouped(axes[1, 1], sub, CLF_MODELS, "test_auc",     "AUC",      f"Test AUC (macro)")

    fig.suptitle(f"Классификация {target_label}: 4 модели × 4 набора признаков", fontsize=12)
    plt.tight_layout()
    path = PLOTS_DIR / f"clf_{pipeline_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Сводка ─────────────────────────────────────────────────────────────────────

def print_summary(df, pipeline, metric_col="test_primary", label=None):
    sub = df[df["pipeline"] == pipeline] if "pipeline" in df.columns else df
    if sub.empty:
        return
    label = label or pipeline
    print(f"\n── {label} (тест SC25) ──")
    for fs in FS_LABELS:
        fsub = sub[sub["feature_set"] == fs].dropna(subset=[metric_col])
        if fsub.empty:
            continue
        idx = fsub[metric_col].idxmin()
        r = fsub.loc[idx]
        extra = ""
        if "test_auc" in r and not np.isnan(r.get("test_auc", np.nan)):
            extra = f"  AUC={r['test_auc']:.3f}"
        if "test_spearman" in r and not np.isnan(r.get("test_spearman", np.nan)):
            extra = f"  R²={r['test_r2']:.2f}  ρ={r['test_spearman']:.2f}"
        print(f"  {fs:<28}  {metric_col}={r[metric_col]:.3f}{extra}  [{r['model']}]")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Загрузка данных...")
    train_df, test_df = load()

    print("\n── Регрессия ──")
    df_reg = run_regression(train_df, test_df)

    print("\n── Классификация Jmax (S-шкала) ──")
    df_jmax_clf = run_jmax_clf(train_df, test_df)

    print("\n── Классификация T_delta ──")
    df_tdelta_clf = run_tdelta_clf(train_df, test_df)

    print("\nПостроение графиков...")
    plot_regression(df_reg)
    plot_classification(df_jmax_clf,   "jmax_clf",   "Jmax S-класс")
    plot_classification(df_tdelta_clf, "tdelta_clf", "T_delta (Быстрые/Умеренные/Медленные)")

    # Сохранение
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_reg.to_excel(writer, sheet_name="regression", index=False)
        df_jmax_clf.to_excel(writer, sheet_name="jmax_clf", index=False)
        df_tdelta_clf.to_excel(writer, sheet_name="tdelta_clf", index=False)
    print(f"\nТаблица: {OUT_XLSX}")

    print_summary(df_reg[df_reg["target"] == "Jmax"],    "regression", label="Регрессия Jmax")
    print_summary(df_reg[df_reg["target"] == "T_delta"], "regression", label="Регрессия T_delta")
    print_summary(df_jmax_clf,   "jmax_clf",   label="Классификация Jmax")
    print_summary(df_tdelta_clf, "tdelta_clf", label="Классификация T_delta")


if __name__ == "__main__":
    main()
