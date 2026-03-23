"""
Классификация СПС по Jmax: категории на основе NOAA S-scale.

  Класс 0 — S1-S2:  10 – 100  pfu
  Класс 1 — S3:    100 – 1000 pfu
  Класс 2 — S4-S5: ≥ 1000    pfu

Все модели выдают вектор вероятностей [P(S1-2), P(S3), P(S4-5)].
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, log_loss,
                             roc_auc_score, confusion_matrix,
                             classification_report)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from spe_utils import build_features, prepare_xy, make_cycle_cv_splits, COL_CYCLE

# ── Константы ─────────────────────────────────────────────────────────────────

CATALOG_PATH = Path(__file__).parent.parent.parent / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx"
SHEET        = "Флюэс GOES"
TRAIN_CYCLES = {23, 24}
TEST_CYCLE   = 25

# NOAA S-scale границы (pfu)
BINS         = [10, 100, 1000, np.inf]
CLASS_LABELS = ["S1-S2\n(10–100)", "S3\n(100–1k)", "S4-S5\n(≥1000)"]
CLASS_SHORT  = ["S1-S2", "S3", "S4-S5"]
N_CLASSES    = 3

CLASS_COLORS = ["#4daf4a", "#ff7f00", "#e41a1c"]   # зелёный, оранжевый, красный

MODEL_COLORS = {
    "LogReg":     "#1f77b4",
    "Forest":     "#8c564b",
    "ExtraTrees": "#c49c94",
    "Boosting":   "#ff7f0e",
    "SVC":        "#2ca02c",
}

KF_STRAT5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── Вспомогательные функции ───────────────────────────────────────────────────

def jmax_to_class(jmax) -> np.ndarray:
    """Jmax (pfu) → номер класса 0/1/2."""
    j = np.asarray(jmax, dtype=float)
    return np.digitize(j, BINS[1:]).astype(int)


def load_data():
    df = pd.read_excel(CATALOG_PATH, sheet_name=SHEET)
    df = build_features(df)
    df = df[df["Jmax"].fillna(0) >= 10].copy()
    df["jmax_class"] = jmax_to_class(df["Jmax"].values)

    train_df = df[df[COL_CYCLE].isin(TRAIN_CYCLES)].copy()
    test_df  = df[df[COL_CYCLE] == TEST_CYCLE].copy()

    print(f"Train SC23+SC24: {len(train_df)}  |  Test SC25: {len(test_df)}  (Jmax>=10)")
    for phase, d in [("Train", train_df), ("Test", test_df)]:
        counts = d["jmax_class"].value_counts().sort_index()
        parts  = [f"{CLASS_SHORT[i]}: {counts.get(i, 0)}" for i in range(N_CLASSES)]
        print(f"  {phase}: {', '.join(parts)}")
    return train_df, test_df


def prepare_clf_xy(df: pd.DataFrame, feature_cols: list):
    """Возвращает X_raw, y_class, valid_idx, cycle_labels."""
    work = df[feature_cols + ["jmax_class"]].copy()
    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    mask = work[feature_cols].apply(np.isfinite).all(axis=1) & work["jmax_class"].notna()
    work = work[mask]
    cycle_labels = (
        pd.to_numeric(df.loc[work.index, COL_CYCLE], errors="coerce").values
        if COL_CYCLE in df.columns else np.full(len(work), np.nan)
    )
    return (work[feature_cols].to_numpy(),
            work["jmax_class"].astype(int).to_numpy(),
            work.index,
            cycle_labels)


# ── Модели ────────────────────────────────────────────────────────────────────

def make_clf_models():
    return {
        "LogReg":     LogisticRegression(
                          max_iter=1000, random_state=42,
                          class_weight="balanced", multi_class="multinomial",
                          solver="lbfgs"),
        "Forest":     RandomForestClassifier(
                          n_estimators=200, random_state=42,
                          class_weight="balanced"),
        "ExtraTrees": ExtraTreesClassifier(
                          n_estimators=200, random_state=42,
                          class_weight="balanced"),
        "Boosting":   GradientBoostingClassifier(
                          n_estimators=200, random_state=42),
        "SVC":        CalibratedClassifierCV(
                          SVC(kernel="rbf", C=10.0,
                              class_weight="balanced",
                              probability=False),
                          cv=3, method="sigmoid"),
    }


# ── Метрики ───────────────────────────────────────────────────────────────────

def brier_multi(y_true, y_prob):
    """Многоклассовый Brier score: mean over classes."""
    y_oh  = label_binarize(y_true, classes=list(range(N_CLASSES)))
    return float(np.mean(np.sum((y_prob - y_oh) ** 2, axis=1)))


def score_clf(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    m = {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "log_loss": round(log_loss(y_true, y_prob, labels=list(range(N_CLASSES))), 3),
        "brier":    round(brier_multi(y_true, y_prob), 3),
    }
    # Macro AUC-ROC (только если все классы представлены)
    try:
        m["auc_macro"] = round(
            roc_auc_score(label_binarize(y_true, classes=list(range(N_CLASSES))),
                          y_prob, multi_class="ovr", average="macro"), 3)
    except Exception:
        m["auc_macro"] = np.nan
    return m


# ── Обучение + оценка ─────────────────────────────────────────────────────────

def clf_fit_and_score(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: list,
    models: dict = None,
) -> dict:
    """
    Обучает классификаторы, считает метрики и вероятности.

    CV-стратегия: cross-cycle (leave-one-cycle-out) если ≥2 цикла,
    иначе StratifiedKFold(5).

    Возвращает dict:
      cv_metrics, test_metrics,
      cv_probs, test_probs, test_true, test_jmax,
      fitted, cycle_names, cv_mode
    """
    if models is None:
        models = make_clf_models()

    X_tr_raw, y_tr, idx_tr, cycle_tr = prepare_clf_xy(train_df, feature_cols)
    X_te_raw, y_te, idx_te, _        = prepare_clf_xy(test_df,  feature_cols)

    if len(X_tr_raw) == 0:
        raise ValueError(f"Train пустой: features={feature_cols}")

    sx = StandardScaler()
    X_tr = sx.fit_transform(X_tr_raw)
    has_test = len(X_te_raw) > 0
    X_te = sx.transform(X_te_raw) if has_test else np.empty((0, X_tr.shape[1]))

    # CV-стратегия
    unique_cycles = sorted(set(c for c in cycle_tr if not np.isnan(c)))
    if len(unique_cycles) >= 2:
        cv_splits   = make_cycle_cv_splits(cycle_tr)
        cycle_names = [f"SC{int(c)}" for c in unique_cycles]
        cv_mode     = "cross_cycle"
    else:
        cv_splits   = list(KF_STRAT5.split(X_tr, y_tr))
        cycle_names = []
        cv_mode     = "stratified_kfold"

    cv_metrics_all   = {}
    test_metrics_all = {}
    cv_probs_all     = {}
    test_probs_all   = {}
    fitted_models    = {}

    for name, mdl in models.items():
        # ── CV: собираем вероятности по фолдам ───────────────────────────────
        probs_cv = np.zeros((len(y_tr), N_CLASSES))

        for tr_idx, val_idx in cv_splits:
            # Проверяем что в фолде есть все классы (важно для Boosting)
            if len(np.unique(y_tr[tr_idx])) < 2:
                continue
            m = clone(mdl)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X_tr[tr_idx], y_tr[tr_idx])
            p = _predict_proba_safe(m, X_tr[val_idx])
            probs_cv[val_idx] = p

        cv_metrics_all[name] = score_clf(y_tr, probs_cv)
        cv_probs_all[name]   = probs_cv

        # ── Полный train → test ───────────────────────────────────────────────
        m_full = clone(mdl)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_full.fit(X_tr, y_tr)
        fitted_models[name] = m_full

        if has_test:
            p_te = _predict_proba_safe(m_full, X_te)
            test_metrics_all[name] = score_clf(y_te, p_te)
            test_probs_all[name]   = p_te
        else:
            test_metrics_all[name] = {k: np.nan for k in
                                      ["accuracy", "log_loss", "brier", "auc_macro"]}
            test_probs_all[name]   = np.zeros((0, N_CLASSES))

    # Jmax и даты тестовых событий (для таблицы прогнозов)
    test_jmax = test_df.loc[idx_te, "Jmax"].values if has_test else np.array([])
    test_dates = (test_df.loc[idx_te, "Дата события"].values
                  if "Дата события" in test_df.columns and has_test else np.array([]))

    return dict(
        cv_metrics=cv_metrics_all,
        test_metrics=test_metrics_all,
        cv_probs=cv_probs_all,
        test_probs=test_probs_all,
        cv_true=y_tr,
        test_true=y_te if has_test else np.array([]),
        test_jmax=test_jmax,
        test_dates=test_dates,
        fitted=fitted_models,
        has_test=has_test,
        cycle_names=cycle_names,
        cv_mode=cv_mode,
        sx=sx,
    )


def _predict_proba_safe(model, X) -> np.ndarray:
    """predict_proba с fallback и выравниванием до N_CLASSES колонок."""
    p = model.predict_proba(X)
    if p.shape[1] == N_CLASSES:
        return p
    # Некоторые классы отсутствовали при обучении → дополняем нулями
    classes = model.classes_
    out = np.zeros((len(X), N_CLASSES))
    for j, c in enumerate(classes):
        if 0 <= c < N_CLASSES:
            out[:, c] = p[:, j]
    # Нормируем строки
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return out / row_sum


# ── Графики ───────────────────────────────────────────────────────────────────

def plot_confusion_matrices(result: dict, title: str, out_path: Path, phase: str = "test"):
    """Матрицы ошибок (нормализованные) для всех моделей."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if phase == "test":
        y_true  = result["test_true"]
        probs   = result["test_probs"]
        has     = result["has_test"]
    else:
        y_true  = result["cv_true"]
        probs   = result["cv_probs"]
        has     = True

    if not has or len(y_true) == 0:
        print(f"  [skip] {out_path.name}")
        return

    models = list(probs.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for ax, name in zip(axes[0], models):
        y_pred = np.argmax(probs[name], axis=1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)),
                              normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f"{cm[i, j]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_SHORT, fontsize=8)
        ax.set_yticklabels(CLASS_SHORT, fontsize=8)
        ax.set_xlabel("Прогноз", fontsize=8)
        ax.set_ylabel("Факт", fontsize=8)
        m = result[f"{phase}_metrics"][name]
        ax.set_title(f"{name}\nAcc={m['accuracy']:.0%}  LogLoss={m['log_loss']:.3f}",
                     fontsize=8)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_roc_curves(result: dict, title: str, out_path: Path, phase: str = "test"):
    """ROC-кривые (one-vs-rest) для каждой модели."""
    from sklearn.metrics import roc_curve
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = result["test_true"] if phase == "test" else result["cv_true"]
    probs  = result["test_probs"] if phase == "test" else result["cv_probs"]
    has    = result["has_test"] if phase == "test" else True

    if not has or len(y_true) == 0:
        print(f"  [skip] {out_path.name}")
        return

    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    models = list(probs.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)

    for ax, name in zip(axes[0], models):
        p = probs[name]
        for k, (lbl, col) in enumerate(zip(CLASS_SHORT, CLASS_COLORS)):
            if y_bin[:, k].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, k], p[:, k])
            auc = roc_auc_score(y_bin[:, k], p[:, k])
            ax.plot(fpr, tpr, color=col, lw=1.5, label=f"{lbl} (AUC={auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_prob_heatmap(result: dict, title: str, out_path: Path):
    """
    Тепловая карта вероятностей по тестовым событиям.
    Строки = события (отсортированы по Jmax), столбцы = P(класс), панели = модели.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not result["has_test"] or len(result["test_true"]) == 0:
        print(f"  [skip] {out_path.name}")
        return

    y_true  = result["test_true"]
    jmax    = result["test_jmax"]
    probs   = result["test_probs"]
    models  = list(probs.keys())
    sort_idx = np.argsort(jmax)

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, max(6, len(y_true) * 0.25 + 2)),
                             squeeze=False)

    for ax, name in zip(axes[0], models):
        p = probs[name][sort_idx]          # (n_events, 3)
        y = y_true[sort_idx]
        j = jmax[sort_idx]

        im = ax.imshow(p, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                       interpolation="nearest")
        for i in range(len(y)):
            for k in range(N_CLASSES):
                clr = "white" if p[i, k] > 0.65 else "black"
                ax.text(k, i, f"{p[i,k]:.2f}", ha="center", va="center",
                        fontsize=5.5, color=clr)
            # Метка реального класса слева
            is_correct = np.argmax(p[i]) == y[i]
            marker = "✓" if is_correct else "✗"
            ax.text(-0.6, i, marker, ha="center", va="center", fontsize=7,
                    color="green" if is_correct else "red")

        ax.set_xticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_SHORT, fontsize=7)
        ax.set_yticks(range(len(y)))
        ax.set_yticklabels([f"{v:.0f}" for v in j], fontsize=5.5)
        ax.set_ylabel("Jmax (pfu)", fontsize=7)
        m = result["test_metrics"][name]
        ax.set_title(f"{name}\nAcc={m['accuracy']:.0%}", fontsize=8)

    fig.suptitle(title, fontsize=10)
    plt.colorbar(im, ax=axes[0][-1], pad=0.02, label="P(класс)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_calibration_clf(result: dict, title: str, out_path: Path):
    """
    Диаграмма калибровки: наблюдаемая доля класса vs. средняя предсказанная вероятность.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = result["cv_true"]
    probs  = result["cv_probs"]
    models = list(probs.keys())
    n_bins = 5

    fig, axes = plt.subplots(1, N_CLASSES,
                             figsize=(5 * N_CLASSES, 4.5), squeeze=False)

    for k, (ax, cls_lbl) in enumerate(zip(axes[0], CLASS_SHORT)):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Идеал")
        for name in models:
            p = probs[name][:, k]
            y_bin = (y_true == k).astype(int)
            # Группируем по бинам вероятности
            bins = np.linspace(0, 1, n_bins + 1)
            mid, frac = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (p >= lo) & (p < hi)
                if mask.sum() >= 2:
                    mid.append(p[mask].mean())
                    frac.append(y_bin[mask].mean())
            if mid:
                ax.plot(mid, frac, "o-",
                        color=MODEL_COLORS.get(name, "#333"),
                        ms=5, lw=1.5, label=name)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Средняя P(класс)")
        ax.set_ylabel("Наблюдаемая доля")
        ax.set_title(f"Класс {cls_lbl}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
