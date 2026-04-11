"""
compare_ew.py
=============
Повторяет пайплайн compare_simple.py с разделением на западные (W) и
восточные (E) события по гелиографической долготе источника вспышки.

Западные:  helio_lon > 0  (W — хорошо связанные)
Восточные: helio_lon < 0  (E — плохо связанные)

Для каждой группы обучаются отдельные модели:
  Train SC23+SC24 [West|East] → Test SC25 [West|East]

Результаты сохраняются в results_ew/:
  plots/   — графики (bar + scatter + importance)
  ew_results.xlsx — числовые метрики

Запуск: python compare_ew.py
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
PLOTS_DIR = ROOT / "results_ew" / "plots"
OUT_XLSX  = ROOT / "results_ew" / "ew_results.xlsx"

# ── Конфигурация (те же наборы признаков, что в compare_simple.py) ────────────

FEATURE_SETS = [
    ("Базовая",           ["helio_lon", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Флюэс вместо пика", ["helio_lon", "log_fluence",        "log_cme_velocity"]),
    ("Обе координаты",    ["helio_lon", "helio_lat", "log_goes_peak_flux", "log_cme_velocity"]),
    ("Координаты+флюэс",  ["helio_lon", "helio_lat", "log_fluence",        "log_cme_velocity"]),
]

REG_MODELS = ["Linear", "Forest", "Boosting", "SVR"]
CLF_MODELS = ["LogReg", "Forest", "Boosting", "SVC"]

FS_LABELS = [fs for fs, _ in FEATURE_SETS]

_CLF_COLORS = {"LogReg": "#1f77b4", "Forest": "#8c564b",
               "Boosting": "#ff7f0e", "SVC": "#2ca02c"}
_ALL_COLORS = {**MODEL_COLORS, **_CLF_COLORS}


# ── Загрузка и разбивка данных ────────────────────────────────────────────────

def load_and_split():
    df = build_features(
        pd.read_excel(ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx",
                      sheet_name="Флюэс GOES")
    )
    cycle = pd.to_numeric(df[COL_CYCLE], errors="coerce")
    df_full = df[(df["Jmax"].fillna(0) >= 10)].copy()

    train_all = df_full[cycle.isin([23, 24])].copy()
    test_all  = df_full[cycle.isin([25])].copy()

    west_mask_tr = train_all["helio_lon"] > 0
    east_mask_tr = train_all["helio_lon"] < 0
    west_mask_te = test_all["helio_lon"]  > 0
    east_mask_te = test_all["helio_lon"]  < 0

    splits = {
        "West": (train_all[west_mask_tr].copy(), test_all[west_mask_te].copy()),
        "East": (train_all[east_mask_tr].copy(), test_all[east_mask_te].copy()),
    }

    for group, (tr, te) in splits.items():
        print(f"  {group}: Train={len(tr)}  Test={len(te)}")

    return splits


# ── Регрессия ─────────────────────────────────────────────────────────────────

def run_regression(train_df, test_df, group: str):
    rows = []
    for fs_label, fs_cols in FEATURE_SETS:
        for tgt_col, tgt_log in [("Jmax", True), ("T_delta", False)]:
            print(f"    [{group}/reg/{tgt_col}] {fs_label} ...", end=" ", flush=True)
            try:
                res = fit_and_score(train_df, test_df, fs_cols, tgt_col, tgt_log)
                for model in REG_MODELS:
                    cv_m = res["cv_metrics"].get(model, {})
                    te_m = res["test_metrics"].get(model, {})
                    primary_cv = cv_m.get("RMSLE_log10", cv_m.get("RMSE", np.nan)) if tgt_log else cv_m.get("RMSE", np.nan)
                    primary_te = te_m.get("RMSLE_log10", te_m.get("RMSE", np.nan)) if tgt_log else te_m.get("RMSE", np.nan)
                    rows.append(dict(
                        group=group, pipeline="regression", target=tgt_col,
                        feature_set=fs_label, model=model,
                        cv_primary=primary_cv, test_primary=primary_te,
                        test_r2=te_m.get("R2_log", te_m.get("R2", np.nan)),
                        test_spearman=te_m.get("Spearman", np.nan),
                    ))
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
    return pd.DataFrame(rows)


# ── Классификация Jmax ────────────────────────────────────────────────────────

def run_jmax_clf(train_df, test_df, group: str):
    from pipelines.classification.utils import make_clf_models, clf_fit_and_score, jmax_to_class
    all_models = make_clf_models()
    models = {k: v for k, v in all_models.items() if k in CLF_MODELS}

    train_df = train_df.copy(); test_df = test_df.copy()
    train_df["jmax_class"] = jmax_to_class(train_df["Jmax"].values)
    test_df["jmax_class"]  = jmax_to_class(test_df["Jmax"].values)

    rows = []
    for fs_label, fs_cols in FEATURE_SETS:
        print(f"    [{group}/jmax_clf] {fs_label} ...", end=" ", flush=True)
        try:
            result = clf_fit_and_score(train_df, test_df, fs_cols, models=models)
            for model in CLF_MODELS:
                cv_m   = result["cv_metrics"].get(model, {})
                test_m = result["test_metrics"].get(model, {})
                rows.append(dict(
                    group=group, pipeline="jmax_clf", target="Jmax_class",
                    feature_set=fs_label, model=model,
                    cv_primary=cv_m.get("log_loss", np.nan),
                    test_primary=test_m.get("log_loss", np.nan),
                    test_accuracy=test_m.get("accuracy", np.nan),
                    test_auc=test_m.get("auc_macro", np.nan),
                ))
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
    return pd.DataFrame(rows)


# ── Классификация T_delta ─────────────────────────────────────────────────────

def run_tdelta_clf(train_df, test_df, group: str):
    from pipelines.tdelta_clf.utils import make_clf_models, clf_fit_and_score, tdelta_to_class
    all_models = make_clf_models()
    models = {k: v for k, v in all_models.items() if k in CLF_MODELS}

    train_df = train_df[train_df["T_delta"].notna()].copy()
    test_df  = test_df[test_df["T_delta"].notna()].copy()
    train_df["tdelta_class"] = tdelta_to_class(train_df["T_delta"].values)
    test_df["tdelta_class"]  = tdelta_to_class(test_df["T_delta"].values)

    rows = []
    for fs_label, fs_cols in FEATURE_SETS:
        print(f"    [{group}/tdelta_clf] {fs_label} ...", end=" ", flush=True)
        try:
            result = clf_fit_and_score(train_df, test_df, fs_cols, models=models)
            for model in CLF_MODELS:
                cv_m   = result["cv_metrics"].get(model, {})
                test_m = result["test_metrics"].get(model, {})
                rows.append(dict(
                    group=group, pipeline="tdelta_clf", target="T_delta_class",
                    feature_set=fs_label, model=model,
                    cv_primary=cv_m.get("log_loss", np.nan),
                    test_primary=test_m.get("log_loss", np.nan),
                    test_accuracy=test_m.get("accuracy", np.nan),
                    test_auc=test_m.get("auc_macro", np.nan),
                ))
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
    return pd.DataFrame(rows)


# ── Визуализация ──────────────────────────────────────────────────────────────

GROUP_COLORS = {"West": "#FF5722", "East": "#2196F3"}


def _bar_grouped_ew(ax, df, model_list, metric_col, ylabel, title):
    """Сгруппированные бары: для каждого набора признаков — West и East рядом."""
    groups  = ["West", "East"]
    n_fs    = len(FS_LABELS)
    n_g     = len(groups)
    bar_w   = 0.35
    xg      = np.arange(n_fs)

    for gi, grp in enumerate(groups):
        sub = df[df["group"] == grp]
        # лучшая модель по метрике для каждого набора признаков
        vals = []
        for fs in FS_LABELS:
            fsub = sub[(sub["feature_set"] == fs)].dropna(subset=[metric_col])
            if fsub.empty:
                vals.append(np.nan)
            else:
                vals.append(fsub[metric_col].min())
        offset = (gi - (n_g - 1) / 2) * bar_w
        ax.bar(xg + offset, vals, width=bar_w,
               color=GROUP_COLORS[grp], label=grp,
               edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_xticks(xg)
    ax.set_xticklabels(FS_LABELS, fontsize=8, rotation=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)


def plot_regression_ew(df):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for tgt, log_tgt in [("Jmax", True), ("T_delta", False)]:
        sub    = df[df["target"] == tgt]
        ylabel = "RMSLE log₁₀ (↓)" if log_tgt else "RMSE ч (↓)"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        _bar_grouped_ew(axes[0], sub, REG_MODELS, "cv_primary",   ylabel, f"CV — {tgt}")
        _bar_grouped_ew(axes[1], sub, REG_MODELS, "test_primary", ylabel, f"Test SC25 — {tgt}")
        fig.suptitle(f"Регрессия {tgt}: West vs East (лучшая модель по набору признаков)",
                     fontsize=12)
        plt.tight_layout()
        path = PLOTS_DIR / f"reg_{tgt.lower()}_ew.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def plot_clf_ew(df, pipeline_name, target_label):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sub = df[df["pipeline"] == pipeline_name]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _bar_grouped_ew(axes[0], sub, CLF_MODELS, "test_primary", "Log Loss (↓)",
                    f"Test Log Loss")
    _bar_grouped_ew(axes[1], sub, CLF_MODELS, "test_auc",     "AUC (↑)",
                    f"Test AUC (macro)")
    fig.suptitle(f"Классификация {target_label}: West vs East", fontsize=12)
    plt.tight_layout()
    path = PLOTS_DIR / f"clf_{pipeline_name}_ew.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Сводка ────────────────────────────────────────────────────────────────────

def print_summary_ew(df, pipeline, target, metric_col="test_primary"):
    sub = df[(df["pipeline"] == pipeline) & (df["target"] == target)] \
        if "target" in df.columns \
        else df[df["pipeline"] == pipeline]
    if sub.empty:
        return
    print(f"\n── {pipeline} / {target} ──")
    print(f"  {'Набор признаков':<28}  {'West':>10}  {'East':>10}  {'Лучший':>8}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*8}")
    for fs in FS_LABELS:
        row_w = sub[(sub["group"] == "West") & (sub["feature_set"] == fs)].dropna(subset=[metric_col])
        row_e = sub[(sub["group"] == "East") & (sub["feature_set"] == fs)].dropna(subset=[metric_col])
        val_w = row_w[metric_col].min() if not row_w.empty else float("nan")
        val_e = row_e[metric_col].min() if not row_e.empty else float("nan")
        better = "West" if val_w < val_e else "East" if val_e < val_w else "—"
        print(f"  {fs:<28}  {val_w:>10.3f}  {val_e:>10.3f}  {better:>8}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Загрузка данных и разбивка East/West...")
    splits = load_and_split()

    all_reg, all_jcf, all_tcf = [], [], []

    for group, (train_df, test_df) in splits.items():
        print(f"\n══ {group} ══")

        print("  Регрессия...")
        all_reg.append(run_regression(train_df, test_df, group))

        print("  Классификация Jmax...")
        all_jcf.append(run_jmax_clf(train_df, test_df, group))

        print("  Классификация T_delta...")
        all_tcf.append(run_tdelta_clf(train_df, test_df, group))

    df_reg = pd.concat(all_reg, ignore_index=True)
    df_jcf = pd.concat(all_jcf, ignore_index=True)
    df_tcf = pd.concat(all_tcf, ignore_index=True)

    print("\nПостроение графиков...")
    plot_regression_ew(df_reg)
    plot_clf_ew(df_jcf, "jmax_clf",   "Jmax S-класс")
    plot_clf_ew(df_tcf, "tdelta_clf", "T_delta (Быстрые/Умеренные/Медленные)")

    # Сохранение
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_reg.to_excel(writer, sheet_name="regression", index=False)
        df_jcf.to_excel(writer, sheet_name="jmax_clf",   index=False)
        df_tcf.to_excel(writer, sheet_name="tdelta_clf", index=False)
    print(f"\nТаблица: {OUT_XLSX}")

    # Сводные таблицы
    print_summary_ew(df_reg, "regression", "Jmax")
    print_summary_ew(df_reg, "regression", "T_delta")
    print_summary_ew(df_jcf, "jmax_clf",   "Jmax_class")
    print_summary_ew(df_tcf, "tdelta_clf", "T_delta_class")


if __name__ == "__main__":
    main()
