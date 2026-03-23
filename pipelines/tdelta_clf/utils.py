"""
Классификация СПС по T_delta (время от начала события до пика потока).

Классы (физически мотивированные):
  0 — Быстрые:    T_delta <  8 ч   (~35% train) — хорошо связанные, импульсные
  1 — Умеренные:  8 ≤ T_delta < 20 ч (~39% train) — стандартные постепенные
  2 — Медленные:  T_delta ≥ 20 ч   (~28% train) — плохая связность / диффузия
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
                             roc_auc_score, confusion_matrix)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spe_utils import build_features, prepare_xy, make_cycle_cv_splits, COL_CYCLE

# ── Константы ─────────────────────────────────────────────────────────────────

CATALOG_PATH = Path(__file__).parent.parent.parent / "data" / "ОБЪЕДИНЕННЫЙ КАТАЛОГ СПС 23-25.xlsx"
SHEET        = "Флюэс GOES"
TRAIN_CYCLES = {23, 24}
TEST_CYCLE   = 25

# Бины T_delta (часы)
BINS_H       = [0, 8, 20, np.inf]
CLASS_LABELS = ["Быстрые\n(< 8 ч)", "Умеренные\n(8–20 ч)", "Медленные\n(≥ 20 ч)"]
CLASS_SHORT  = ["Быстрые", "Умеренные", "Медленные"]
N_CLASSES    = 3

CLASS_COLORS = ["#4daf4a", "#ff7f00", "#e41a1c"]

MODEL_COLORS = {
    "LogReg":     "#1f77b4",
    "Forest":     "#8c564b",
    "ExtraTrees": "#c49c94",
    "Boosting":   "#ff7f0e",
    "SVC":        "#2ca02c",
}

KF_STRAT5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── Вспомогательные функции ───────────────────────────────────────────────────

def tdelta_to_class(tdelta) -> np.ndarray:
    """T_delta (часы) → номер класса 0/1/2."""
    t = np.asarray(tdelta, dtype=float)
    cls = np.zeros(len(t), dtype=int)
    cls[(t >= BINS_H[1]) & (t < BINS_H[2])] = 1
    cls[t >= BINS_H[2]] = 2
    return cls


def load_data():
    df = pd.read_excel(CATALOG_PATH, sheet_name=SHEET)
    df = build_features(df)
    df = df[df["Jmax"].fillna(0) >= 10].copy()
    df = df[df["T_delta"].notna()].copy()
    df["tdelta_class"] = tdelta_to_class(df["T_delta"].values)

    train_df = df[df[COL_CYCLE].isin(TRAIN_CYCLES)].copy()
    test_df  = df[df[COL_CYCLE] == TEST_CYCLE].copy()

    print(f"Train SC23+SC24: {len(train_df)}  |  Test SC25: {len(test_df)}  (Jmax>=10, T_delta не NaN)")
    for phase, d in [("Train", train_df), ("Test", test_df)]:
        counts = d["tdelta_class"].value_counts().sort_index()
        parts  = [f"{CLASS_SHORT[i]}: {counts.get(i, 0)}" for i in range(N_CLASSES)]
        print(f"  {phase}: {', '.join(parts)}")
    return train_df, test_df


def prepare_clf_xy(df: pd.DataFrame, feature_cols: list):
    """Возвращает X_raw, y_class, valid_idx, cycle_labels."""
    work = df[feature_cols + ["tdelta_class"]].copy()
    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    mask = work[feature_cols].apply(np.isfinite).all(axis=1) & work["tdelta_class"].notna()
    work = work[mask]
    cycle_labels = (
        pd.to_numeric(df.loc[work.index, COL_CYCLE], errors="coerce").values
        if COL_CYCLE in df.columns else np.full(len(work), np.nan)
    )
    return (work[feature_cols].to_numpy(),
            work["tdelta_class"].astype(int).to_numpy(),
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
    y_oh = label_binarize(y_true, classes=list(range(N_CLASSES)))
    return float(np.mean(np.sum((y_prob - y_oh) ** 2, axis=1)))


def score_clf(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    m = {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "log_loss": round(log_loss(y_true, y_prob, labels=list(range(N_CLASSES))), 3),
        "brier":    round(brier_multi(y_true, y_prob), 3),
    }
    try:
        m["auc_macro"] = round(
            roc_auc_score(label_binarize(y_true, classes=list(range(N_CLASSES))),
                          y_prob, multi_class="ovr", average="macro"), 3)
    except Exception:
        m["auc_macro"] = np.nan
    return m


# ── Обучение + оценка ─────────────────────────────────────────────────────────

def clf_fit_and_score(train_df, test_df, feature_cols, models=None):
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
        probs_cv = np.full((len(y_tr), N_CLASSES), 1 / N_CLASSES)

        for tr_idx, val_idx in cv_splits:
            if len(np.unique(y_tr[tr_idx])) < 2:
                continue
            m = clone(mdl)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X_tr[tr_idx], y_tr[tr_idx])
            probs_cv[val_idx] = _predict_proba_safe(m, X_tr[val_idx])

        cv_metrics_all[name] = score_clf(y_tr, probs_cv)
        cv_probs_all[name]   = probs_cv

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

    tdelta_te = test_df.loc[idx_te, "T_delta"].values if has_test else np.array([])

    return dict(
        cv_metrics=cv_metrics_all,
        test_metrics=test_metrics_all,
        cv_probs=cv_probs_all,
        test_probs=test_probs_all,
        cv_true=y_tr,
        test_true=y_te if has_test else np.array([]),
        tdelta_te=tdelta_te,
        fitted=fitted_models,
        has_test=has_test,
        cycle_names=cycle_names,
        cv_mode=cv_mode,
        sx=sx,
    )


def _predict_proba_safe(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    if p.shape[1] == N_CLASSES:
        return p
    classes = model.classes_
    out = np.zeros((len(X), N_CLASSES))
    for j, c in enumerate(classes):
        if 0 <= c < N_CLASSES:
            out[:, c] = p[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return out / row_sum


# ── Графики ───────────────────────────────────────────────────────────────────

def plot_confusion_matrices(result: dict, title: str, out_path: Path, phase: str = "test"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = result["test_true"] if phase == "test" else result["cv_true"]
    probs  = result["test_probs"] if phase == "test" else result["cv_probs"]
    has    = result["has_test"] if phase == "test" else True

    if not has or len(y_true) == 0:
        return

    models = list(probs.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)

    for ax, name in zip(axes[0], models):
        y_pred = np.argmax(probs[name], axis=1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)), normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_SHORT, fontsize=7, rotation=15)
        ax.set_yticklabels(CLASS_SHORT, fontsize=7)
        ax.set_xlabel("Прогноз", fontsize=8); ax.set_ylabel("Факт", fontsize=8)
        m = result[f"{phase}_metrics"][name]
        ax.set_title(f"{name}\nAcc={m['accuracy']:.0%}  LL={m['log_loss']:.3f}", fontsize=8)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
