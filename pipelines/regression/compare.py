"""
Сравнительный анализ наборов входных признаков.

Для каждого набора признаков обучаются те же 6 моделей, результаты
сравниваются по CV RMSE (train SC23+SC24) и Test RMSE (SC25).

Запуск: python compare.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from spe_utils import (
    build_features, fit_and_score,
    COL_CYCLE, COL_JMAX, COL_TDELTA,
    MODEL_COLORS,
)

ROOT       = Path(__file__).parent.parent.parent
PIPELINE   = Path(__file__).parent
INPUT_XLSX = ROOT / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx"
SHEET      = "Флюэс GOES"
PLOTS_DIR  = PIPELINE / "plots" / "comparison"
OUT_XLSX   = PIPELINE / "results" / "comparison_results.xlsx"

# ─── Конфигурации наборов признаков ──────────────────────────────────────────
# Каждый: (label, [feature_cols])
# log_flare_power и log_cme_velocity уже добавлены в build_features().
# Используем log10-версии широкодиапазонных признаков для улучшения линейных моделей.
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
    dict(col="Jmax",    label="Jmax (pfu)",       log=True,  subdir="jmax"),
    dict(col="T_delta", label="T_delta (часы)",    log=False, subdir="tdelta"),
]

MODEL_ORDER = ["Linear", "Ridge", "Huber", "Forest", "ExtraTrees", "Boosting", "SVR", "GPR_RBF"]


# ─── Сбор метрик ─────────────────────────────────────────────────────────────

def _primary_metric(metrics_dict: dict, log_target: bool) -> float:
    """Основная метрика: RMSLE_log10 для Jmax, RMSE для T_delta."""
    if log_target:
        return metrics_dict.get("RMSLE_log10", metrics_dict.get("RMSE", np.nan))
    return metrics_dict.get("RMSE", np.nan)


def collect_metrics(train_df, test_df):
    """
    Собирает метрики для всех комбинаций (target x feature_set x model).
    Основная метрика: RMSLE_log10 (Jmax) или RMSE (T_delta).
    """
    records = []
    for tgt in TARGETS:
        for fs_label, fs_cols in FEATURE_SETS:
            print(f"  [{tgt['col']}] {fs_label.replace(chr(10),' ')}", end=" ... ", flush=True)
            try:
                res = fit_and_score(train_df, test_df, fs_cols, tgt["col"], tgt["log"])
                for model in MODEL_ORDER:
                    cv_m  = res["cv_metrics"].get(model, {})
                    te_m  = res["test_metrics"].get(model, {})
                    records.append({
                        "target":         tgt["col"],
                        "target_label":   tgt["label"],
                        "feature_set":    fs_label,
                        "model":          model,
                        "cv_rmse":        cv_m.get("RMSE", np.nan),
                        "test_rmse":      te_m.get("RMSE", np.nan),
                        "cv_primary":     _primary_metric(cv_m,  tgt["log"]),
                        "test_primary":   _primary_metric(te_m,  tgt["log"]),
                        "test_r2":        te_m.get("R2_log", te_m.get("R2", np.nan)),
                        "test_spearman":  te_m.get("Spearman", np.nan),
                    })
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")

    return pd.DataFrame(records)


# ─── Визуализация ─────────────────────────────────────────────────────────────

def _heatmap(ax, data_2d, row_labels, col_labels, title, fmt=".1f", cmap="YlOrRd"):
    """data_2d: (n_rows, n_cols)"""
    im = ax.imshow(data_2d, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for r in range(data_2d.shape[0]):
        for c in range(data_2d.shape[1]):
            v = data_2d[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=7.5, color="black")
    ax.set_title(title, fontsize=10)
    return im


def plot_heatmaps(df_metrics: pd.DataFrame, out_dir: Path):
    """
    Тепловые карты: строки = feature_set, столбцы = модель.
    Jmax: основная метрика = RMSLE_log10 (ниже = лучше).
    T_delta: RMSE (ниже = лучше). Плюс R2 и Spearman на тесте.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fs_labels = [fs for fs, _ in FEATURE_SETS]

    for tgt in TARGETS:
        sub = df_metrics[df_metrics["target"] == tgt["col"]]
        cv_mat    = np.full((len(FEATURE_SETS), len(MODEL_ORDER)), np.nan)
        test_mat  = np.full((len(FEATURE_SETS), len(MODEL_ORDER)), np.nan)
        r2_mat    = np.full((len(FEATURE_SETS), len(MODEL_ORDER)), np.nan)

        for ri, (fs_label, _) in enumerate(FEATURE_SETS):
            for ci, model in enumerate(MODEL_ORDER):
                row = sub[(sub["feature_set"] == fs_label) & (sub["model"] == model)]
                if not row.empty:
                    cv_mat[ri, ci]   = row["cv_primary"].values[0]
                    test_mat[ri, ci] = row["test_primary"].values[0]
                    r2_mat[ri, ci]   = row["test_r2"].values[0]

        primary_label = "RMSLE log10 (ниже лучше)" if tgt["log"] else "RMSE (ниже лучше)"
        row_labels_short = [fs.replace("\n", " ") for fs in fs_labels]

        fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
        _heatmap(axes[0], cv_mat,   row_labels_short, MODEL_ORDER, f"CV {primary_label} (train)")
        _heatmap(axes[1], test_mat, row_labels_short, MODEL_ORDER, f"Test {primary_label} (SC25)")
        _heatmap(axes[2], r2_mat,   row_labels_short, MODEL_ORDER, "R2_log (test)", fmt=".2f", cmap="RdYlGn")

        fig.suptitle(f"Сравнение наборов признаков -- {tgt['label']}", fontsize=12)
        plt.tight_layout()
        path = out_dir / f"heatmap_{tgt['subdir']}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def plot_bars(df_metrics: pd.DataFrame, out_dir: Path, metric: str = "test_primary"):
    """
    Сгруппированные столбчатые графики.
    Наборы признаков разбиваются на строки по CHUNK штук, стакуются вертикально.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fs_labels = [fs for fs, _ in FEATURE_SETS]
    n_fs  = len(fs_labels)
    n_m   = len(MODEL_ORDER)
    CHUNK = 5  # наборов признаков на одну строку

    chunks = [list(range(i, min(i + CHUNK, n_fs))) for i in range(0, n_fs, CHUNK)]
    n_rows = len(chunks)

    for tgt in TARGETS:
        sub = df_metrics[df_metrics["target"] == tgt["col"]]
        primary_label = "RMSLE_log10" if tgt["log"] else "RMSE"
        metric_title  = "Test (SC25)" if "test" in metric else "CV (train)"

        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4.5 * n_rows), squeeze=False)

        for row_idx, chunk_idxs in enumerate(chunks):
            ax = axes[row_idx, 0]
            n_chunk = len(chunk_idxs)
            bar_w = 0.7 / n_m
            xg = np.arange(n_chunk)

            for mi, model in enumerate(MODEL_ORDER):
                offset = (mi - (n_m - 1) / 2) * bar_w
                vals = []
                for fi in chunk_idxs:
                    fs_label = fs_labels[fi]
                    row = sub[(sub["feature_set"] == fs_label) & (sub["model"] == model)]
                    vals.append(row[metric].values[0] if not row.empty else np.nan)
                ax.bar(xg + offset, vals, width=bar_w,
                       color=MODEL_COLORS.get(model, "#888"),
                       label=model, edgecolor="white", linewidth=0.5)

            ax.set_xticks(xg)
            ax.set_xticklabels([fs_labels[fi] for fi in chunk_idxs], fontsize=9)
            ax.set_ylabel(primary_label)
            ax.grid(axis="y", alpha=0.3)
            if row_idx == 0:
                ax.legend(ncol=4, fontsize=9)

        fig.suptitle(f"{metric_title} {primary_label} — {tgt['label']}", fontsize=12)
        plt.tight_layout()
        path = out_dir / f"bars_{metric}_{tgt['subdir']}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def plot_best_model_comparison(df_metrics: pd.DataFrame, out_dir: Path):
    """
    Для каждого набора признаков — лучшая модель по test RMSE.
    Линейный график «feature_set vs min RMSE», отдельно для каждого target.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fs_labels = [fs.replace("\n", " ") for fs, _ in FEATURE_SETS]

    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(14, 5 * len(TARGETS)))
    if len(TARGETS) == 1:
        axes = [axes]

    for ax, tgt in zip(axes, TARGETS):
        sub = df_metrics[df_metrics["target"] == tgt["col"]]
        best_val, best_model = [], []
        for fs_label, _ in FEATURE_SETS:
            fs_sub = sub[sub["feature_set"] == fs_label].dropna(subset=["test_primary"])
            if fs_sub.empty:
                best_val.append(np.nan)
                best_model.append("N/A")
                continue
            idx = fs_sub["test_primary"].idxmin()
            best_val.append(fs_sub.loc[idx, "test_primary"])
            best_model.append(fs_sub.loc[idx, "model"])

        x = range(len(FEATURE_SETS))
        valid = [v for v in best_val if not (isinstance(v, float) and np.isnan(v))]
        ax.plot(x, best_val, "o-", color="#e74c3c", lw=2, ms=8)
        for xi, (v, model) in enumerate(zip(best_val, best_model)):
            if isinstance(v, float) and np.isnan(v):
                ax.text(xi, (max(valid) if valid else 1) * 0.5,
                        f"{model}\nN/A", ha="center", fontsize=7, color="gray")
            else:
                ax.text(xi, v + (max(valid) if valid else 1) * 0.01,
                        f"{model}\n{v:.3f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(fs_labels, rotation=15, ha="right", fontsize=8)
        metric_name = "RMSLE_log10" if tgt["log"] else "RMSE"
        ax.set_ylabel(f"Test {metric_name} (SC25)")
        ax.set_title(f"Лучшая модель по набору признаков\n{tgt['label']}", fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "best_model_per_featureset.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── main ─────────────────────────────────────────────────────────────────────

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
    print(f"Train SC23+SC24: {len(train_df)}  |  Test SC25: {len(test_df)}  (Jmax>=10)\n")

    print("Обучение моделей по всем конфигурациям...")
    df_metrics = collect_metrics(train_df, test_df)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nПостроение графиков...")
    plot_heatmaps(df_metrics, PLOTS_DIR)
    plot_bars(df_metrics, PLOTS_DIR, metric="test_primary")
    plot_bars(df_metrics, PLOTS_DIR, metric="cv_primary")
    plot_best_model_comparison(df_metrics, PLOTS_DIR)

    # Сводные таблицы
    for sheet_metric, col in [("test_primary", "test_primary"), ("cv_primary", "cv_primary"),
                               ("test_R2", "test_r2"), ("test_Spearman", "test_spearman")]:
        try:
            piv = df_metrics.pivot_table(
                index=["target_label", "feature_set"], columns="model", values=col
            )[MODEL_ORDER]
            globals()[f"_piv_{sheet_metric}"] = piv
        except Exception:
            globals()[f"_piv_{sheet_metric}"] = pd.DataFrame()

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_metrics.to_excel(writer, sheet_name="raw", index=False)
        for sheet_metric in ["test_primary", "cv_primary", "test_R2", "test_Spearman"]:
            piv = globals()[f"_piv_{sheet_metric}"]
            if not piv.empty:
                piv.to_excel(writer, sheet_name=sheet_metric)

    print(f"\nТаблица результатов: {OUT_XLSX!r}")
    print(f"Графики:            {PLOTS_DIR}/")

    # Краткая сводка
    print("\n== Лучшие результаты по набору признаков (основная метрика) ==")
    for tgt in TARGETS:
        primary_name = "RMSLE_log10" if tgt["log"] else "RMSE"
        print(f"\n{tgt['label']}  [{primary_name}]:")
        sub = df_metrics[df_metrics["target"] == tgt["col"]]
        summary = (
            sub.groupby("feature_set")["test_primary"]
            .min().reset_index().sort_values("test_primary")
        )
        for _, row in summary.iterrows():
            if pd.isna(row["test_primary"]):
                fs_short = row["feature_set"].replace("\n", " ")
                print(f"  {fs_short:<40} {'N/A (нет данных SC25)':>7}")
                continue
            best = sub[
                (sub["feature_set"] == row["feature_set"]) &
                (sub["test_primary"] == row["test_primary"])
            ].dropna(subset=["test_primary"])
            if best.empty:
                continue
            model = best["model"].values[0]
            r2    = best["test_r2"].values[0]
            sp    = best["test_spearman"].values[0]
            fs_short = row["feature_set"].replace("\n", " ")
            print(f"  {fs_short:<40} {row['test_primary']:>7.3f}  R2={r2:.2f}  Sp={sp:.2f}  [{model}]")


if __name__ == "__main__":
    main()
