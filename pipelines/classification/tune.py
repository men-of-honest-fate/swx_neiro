"""
Тюнинг гиперпараметров классификаторов.
Минимизируем CV Log Loss.

Запуск:  python tune.py
"""

import sys
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import (
    load_data, make_clf_models, clf_fit_and_score,
    prepare_clf_xy, make_cycle_cv_splits, CLASS_SHORT, N_CLASSES,
    KF_STRAT5, _predict_proba_safe,
)
from spe_utils import COL_CYCLE

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent.parent
PIPELINE = Path(__file__).parent

CONFIGS = {
    "baseline": ["helio_lon", "log_flare_power", "log_cme_velocity"],
    "best":     ["t_delta_flare", "log_flare_power", "log_cme_velocity"],
}

PARAM_GRIDS = {
    "LogReg": {
        "C":       [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l2"],
    },
    "Forest": {
        "n_estimators":     [100, 200, 400],
        "max_depth":        [None, 5, 10],
        "min_samples_leaf": [1, 3, 5],
    },
    "Boosting": {
        "n_estimators":  [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth":     [3, 5],
    },
    "SVC": {
        "C":       [1.0, 10.0, 100.0],
        "kernel":  ["rbf", "poly"],
    },
}


def _make_model(model_name, params):
    if model_name == "LogReg":
        return LogisticRegression(max_iter=1000, random_state=42,
                                  class_weight="balanced",
                                  multi_class="multinomial", solver="lbfgs",
                                  **params)
    if model_name == "Forest":
        return RandomForestClassifier(random_state=42,
                                      class_weight="balanced", **params)
    if model_name == "Boosting":
        return GradientBoostingClassifier(random_state=42, **params)
    if model_name == "SVC":
        return CalibratedClassifierCV(
            SVC(class_weight="balanced", probability=False, **params),
            cv=3, method="sigmoid")
    raise ValueError(model_name)


def cv_log_loss(train_df, feature_cols, model_name, params) -> float:
    """Быстрый cross-cycle log loss для заданных гиперпараметров."""
    X_raw, y, _, cycle_tr = prepare_clf_xy(train_df, feature_cols)
    sx = StandardScaler()
    X  = sx.fit_transform(X_raw)

    unique_cycles = sorted(set(c for c in cycle_tr if not np.isnan(c)))
    if len(unique_cycles) >= 2:
        splits = make_cycle_cv_splits(cycle_tr)
    else:
        splits = list(KF_STRAT5.split(X, y))

    probs_cv = np.zeros((len(y), N_CLASSES))
    for tr_idx, val_idx in splits:
        if len(np.unique(y[tr_idx])) < 2:
            continue
        mdl = clone(_make_model(model_name, params))
        mdl.fit(X[tr_idx], y[tr_idx])
        probs_cv[val_idx] = _predict_proba_safe(mdl, X[val_idx])

    return log_loss(y, probs_cv, labels=list(range(N_CLASSES)))


def grid_search(train_df, feature_cols, model_name, param_grid):
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    total  = 1
    for v in values: total *= len(v)
    print(f"    {model_name}: {total} комбинаций")

    best_ll = np.inf
    best_p  = {}
    rows    = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        try:
            ll = cv_log_loss(train_df, feature_cols, model_name, params)
        except Exception as e:
            ll = np.nan
        rows.append({**params, "cv_log_loss": ll})
        if ll < best_ll:
            best_ll = ll
            best_p  = params

    df = pd.DataFrame(rows).sort_values("cv_log_loss")
    return best_p, best_ll, df


def main():
    train_df, test_df = load_data()
    all_rows = []

    for cfg_name, feature_cols in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Конфигурация: {cfg_name}  features={feature_cols}")
        print(f"{'='*60}")

        # Базовые метрики
        base = clf_fit_and_score(train_df, test_df, feature_cols)
        print("  Базовый CV Log Loss:")
        for name, m in base["cv_metrics"].items():
            print(f"    {name:<14} {m['log_loss']:.4f}  Acc={m['accuracy']:.0%}")

        # Тюнинг
        for model_name, param_grid in PARAM_GRIDS.items():
            best_p, best_ll, df_grid = grid_search(
                train_df, feature_cols, model_name, param_grid)
            print(f"\n    {model_name} — лучшие параметры: {best_p}")
            print(f"    Log Loss: {base['cv_metrics'].get(model_name,{}).get('log_loss','?')}"
                  f"  →  {best_ll:.4f}")
            top = df_grid.head(3)[["cv_log_loss"] + list(param_grid.keys())]
            print("    Топ-3:", top.to_string(index=False))

            all_rows.append(dict(
                config=cfg_name, model=model_name,
                base_ll=base["cv_metrics"].get(model_name, {}).get("log_loss", np.nan),
                tuned_ll=best_ll, **best_p,
            ))

    df_out = pd.DataFrame(all_rows)
    out_path = PIPELINE / "results" / "tune_results.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(out_path, index=False)
    print(f"\nРезультаты: '{out_path}'")

    print("\n" + "=" * 60)
    print("  Итог тюнинга:")
    for _, row in df_out.iterrows():
        delta = row["base_ll"] - row["tuned_ll"]
        params = {k: v for k, v in row.items()
                  if k not in ("config", "model", "base_ll", "tuned_ll")}
        print(f"  [{row['config']}] {row['model']:<14} "
              f"{row['base_ll']:.4f} → {row['tuned_ll']:.4f}  "
              f"(Δ={delta:+.4f})  {params}")


if __name__ == "__main__":
    main()
