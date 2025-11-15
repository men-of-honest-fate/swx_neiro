import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


INPUT_CSV = "data/processed.csv"
OUTPUT_XLSX = "predictions.xlsx"


def _prepare_xy(df: pd.DataFrame, feature_cols, target_col: str):
    feature_cols = list(feature_cols)
    cols = feature_cols + [target_col]
    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    mask = work[cols].apply(np.isfinite).all(axis=1)
    work = work[mask]
    return work[feature_cols].to_numpy(), work[target_col].to_numpy(), work.index


def permutation_importance_mse_custom(estimator, X, y, scorer, n_repeats=20, random_state=42):
    rng = np.random.default_rng(random_state)
    base_score = scorer(estimator, X, y)
    importances = np.zeros(X.shape[1], dtype=float)

    for j in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            scores.append(scorer(estimator, Xp, y))
        delta = base_score - np.mean(scores)
        importances[j] = max(delta, 0.0)

    return importances


def build_importances_for_target(
    models_dict: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list | tuple,
    *,
    is_jmax: bool = False,
    y_scaler=None,
    inv_target: bool = False,
):
    feature_names = list(feature_names)
    out = {}

    for name, model in models_dict.items():
        fi = None
        if hasattr(model, "feature_importances_"):
            fi = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "named_steps"):
            for _, st in model.named_steps.items():
                if hasattr(st, "feature_importances_"):
                    fi = np.asarray(st.feature_importances_, dtype=float)
                    break
        if fi is None and hasattr(model, "named_steps"):
            steps = list(model.named_steps.items())
            last = steps[-1][1]
            if hasattr(last, "coef_"):
                fi = np.abs(np.asarray(last.coef_, dtype=float))
        if fi is None:
            fi = np.zeros(len(feature_names), dtype=float)

        fi = np.nan_to_num(fi, nan=0.0, posinf=0.0, neginf=0.0)
        s = fi.sum()
        fi_pct = (fi / s * 100.0) if s > 0 else fi
        out[(name, "builtin")] = fi_pct

    for name, model in models_dict.items():
        if is_jmax and inv_target and y_scaler is not None:
            def scorer(est, X, y_true):
                y_pred_scaled = est.predict(X)
                y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                y_pred = np.expm1(y_pred_log)
                return -mean_squared_error(y_true, y_pred)

            pimps = permutation_importance_mse_custom(
                model, X_train, y_train, scorer, n_repeats=50, random_state=42
            )
        else:
            res = permutation_importance(
                model, X_train, y_train,
                n_repeats=50, random_state=42, scoring="neg_mean_squared_error"
            )
            pimps = -np.asarray(res.importances_mean, dtype=float)

        pimps[pimps < 0] = 0.0
        pimps = np.nan_to_num(pimps, nan=0.0, posinf=0.0, neginf=0.0)
        s = pimps.sum()
        pimps_pct = (pimps / s * 100.0) if s > 0 else pimps
        out[(name, "permutation")] = pimps_pct

    try:
        import shap
    except Exception:
        shap = None

    for name, model in models_dict.items():
        shap_imp = np.zeros(len(feature_names), dtype=float)
        if shap is not None:
            base_est = model
            if hasattr(model, "named_steps"):
                for _, st in model.named_steps.items():
                    if hasattr(st, "feature_importances_"):
                        base_est = st
                        break
            try:
                explainer = shap.TreeExplainer(base_est)
                sv = explainer.shap_values(X_train)
            except Exception:
                try:
                    explainer = shap.LinearExplainer(base_est, X_train, feature_perturbation="interventional")
                    sv = explainer.shap_values(X_train)
                except Exception:
                    sv = None
            if sv is not None:
                if isinstance(sv, list):
                    sv = sv[0]
                shap_imp = np.mean(np.abs(np.asarray(sv)), axis=0)

        shap_imp = np.nan_to_num(shap_imp, nan=0.0, posinf=0.0, neginf=0.0)
        s = shap_imp.sum()
        shap_pct = (shap_imp / s * 100.0) if s > 0 else shap_imp
        out[(name, "shap")] = shap_pct

    return out


def plot_separated_method_importances(
    importances_dict: dict,
    feature_names: list | tuple,
    title: str,
    output_path: str,
    decimals: int = 1
):
    feature_names = list(feature_names)
    methods = ["builtin", "permutation", "shap"]
    models = sorted({k[0] for k in importances_dict.keys()})
    n_feats = len(feature_names)
    n_models = len(models)

    mats = {}
    for method in methods:
        mat = np.zeros((n_models, n_feats), dtype=float)
        for mi, m in enumerate(models):
            vec = importances_dict.get((m, method), np.zeros(n_feats, dtype=float))
            mat[mi, :] = vec
        mats[method] = mat

    fig, axes = plt.subplots(
        nrows=3, ncols=1,
        figsize=(max(10, n_models * n_feats * 0.9), 6 + 2.0),
        sharex=False
    )

    group_width = 0.8
    bar_width = group_width / max(n_feats, 1)

    cmap_feats = plt.cm.get_cmap("Set2", n_feats)

    for ax, method in zip(axes, methods):
        mat = mats[method]

        x_groups = np.arange(n_models)

        bars = []

        for fi, feat in enumerate(feature_names):
            offset = (fi - (n_feats - 1) / 2) * bar_width
            x = x_groups + offset
            y = mat[:, fi]
            b = ax.bar(x, y, width=bar_width, color=cmap_feats(fi), label=feat)
            bars.append(b)

            for rect in b:
                h = rect.get_height()
                if h > 0:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        h, f"{h:.{decimals}f}%",
                        ha="center", va="bottom", fontsize=8
                    )

        ax.set_xticks(x_groups)
        ax.set_xticklabels(models)
        ax.set_ylabel("Вклад, %")
        ax.set_title(f"{method}", loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        ax.legend(ncol=min(n_feats, 4), fontsize=9, loc="upper right", title="Признаки")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"График (3 подграфика; группировка по моделям) сохранён в '{output_path}'")
    plt.show()


def main():
    warnings.filterwarnings("ignore")

    header = pd.read_csv(INPUT_CSV, nrows=0)
    parse_dates = [c for c in ["Event_date", "Tmax_parsed"] if c in header.columns]
    df = pd.read_csv(INPUT_CSV, parse_dates=parse_dates)

    required = {"T_delta_flare", "Flare_power", "Jmax_parsed", "T_delta_SPE", "Cycle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют требуемые колонки: {sorted(missing)}")

    cycle_num = pd.to_numeric(df["Cycle"], errors="coerce")
    df = df.assign(cycle_num=cycle_num)

    train_df = df[df["cycle_num"].isin([23, 24])].copy()
    test_df  = df[df["cycle_num"].isin([25])].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Недостаточно данных в train (cycles 23–24) или test (cycle 25).")

    FEATS = ["T_delta_flare", "Flare_power"]

    # Масштабировщики признаков и целей для каждого таргета
    X_scaler_j = StandardScaler()
    y_scaler_j_log = StandardScaler()  # для лог-тренированного Jmax
    
    X_scaler_t = StandardScaler()
    y_scaler_t = StandardScaler()  # для T_delta_SPE

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    base_cols = [c for c in ["Event_date", "T_delta_flare", "Flare_power", "Cycle"] if c in df.columns]
    new_table = df[base_cols].copy()
    new_table["split"] = np.where(df.index.isin(test_df.index), "test_sc25", "train_sc23_24")

    # =========================
    # Блок A: Jmax_parsed (с лог-трансформацией И масштабированием)
    # =========================
    Xtr_j, ytr_j, idx_tr_j = _prepare_xy(train_df, FEATS, "Jmax_parsed")
    
    # Масштабируем признаки
    X_scaled_j = X_scaler_j.fit_transform(Xtr_j)
    
    # Лог-трансформация и масштабирование цели
    ytr_log = np.log1p(ytr_j)
    y_scaled_j = y_scaler_j_log.fit_transform(ytr_log.reshape(-1, 1)).ravel()

    print("\n" + "=" * 40)
    print("Jmax_parsed — Train (cycles 23–24) CV RMSE")
    print("=" * 40)
    print("[Модели обучаются на масштабированных X и y (лог-трансформированные)]")

    for name in ["Forest", "Boosting", "Linear"]:
        new_table[f"{name}_Jpred"] = np.nan

    # Инициализируем модели
    models_j = {
        "Forest": RandomForestRegressor(random_state=42),
        "Boosting": GradientBoostingRegressor(random_state=42),
        "Linear": LinearRegression(),
    }

    # CV на масштабированном пространстве
    for name, model in models_j.items():
        y_cv_pred_scaled = cross_val_predict(model, X_scaled_j, y_scaled_j, cv=kf)
        
        # Обратные преобразования для оценки RMSE в исходной шкале
        y_cv_pred_log = y_scaler_j_log.inverse_transform(y_cv_pred_scaled.reshape(-1, 1)).ravel()
        y_cv_pred = np.expm1(y_cv_pred_log)
        y_cv_pred = np.clip(y_cv_pred, 0.0, None)
        
        rmse_cv = root_mean_squared_error(ytr_j, y_cv_pred)
        print(f"{name}: {rmse_cv:.2f}")

    # Обучение и тестирование
    Xte_j, yte_j, idx_te_j = _prepare_xy(test_df, FEATS, "Jmax_parsed")
    X_scaled_te_j = X_scaler_j.transform(Xte_j)

    for name, model in models_j.items():
        # Обучение на масштабированном пространстве
        model.fit(X_scaled_j, y_scaled_j)
        
        # Предсказание на тесте в масштабированном пространстве
        y_test_pred_scaled = model.predict(X_scaled_te_j)
        
        # Обратные преобразования
        y_test_pred_log = y_scaler_j_log.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
        y_test_pred = np.expm1(y_test_pred_log)
        y_test_pred = np.clip(y_test_pred, 0.0, None)
        
        rmse_test = root_mean_squared_error(yte_j, y_test_pred)
        print(f"Test SC25 {name} RMSE: {rmse_test:.2f}")
        new_table.loc[idx_te_j, f"{name}_Jpred"] = y_test_pred

    # Важности для Jmax
    imp_j = build_importances_for_target(
        models_dict=models_j,
        X_train=X_scaled_j,
        y_train=y_scaled_j,
        feature_names=FEATS,
        is_jmax=True,
        y_scaler=y_scaler_j_log,
        inv_target=True
    )
    plot_separated_method_importances(
        imp_j, FEATS,
        title="Jmax_parsed — Builtin vs Permutation(ΔMSE) vs SHAP",
        output_path="plots/importance_jmax.png",
        decimals=1
    )

    # =========================
    # Блок B: T_delta_SPE (с масштабированием y)
    # =========================
    Xtr_t, ytr_t, idx_tr_t = _prepare_xy(train_df, FEATS, "T_delta_SPE")
    
    # Масштабируем X и y
    X_scaled_t = X_scaler_t.fit_transform(Xtr_t)
    y_scaled_t = y_scaler_t.fit_transform(ytr_t.reshape(-1, 1)).ravel()

    print("\n" + "=" * 40)
    print("Delta_T_max (T_delta_SPE) — Train (cycles 23–24) CV RMSE")
    print("=" * 40)
    print("[Модели обучаются на масштабированных X и y]")

    for name in ["Forest", "Boosting", "Linear"]:
        col = f"{name}_Delta_T_max"
        if col not in new_table.columns:
            new_table[col] = np.nan

    models_t = {
        "Forest": RandomForestRegressor(random_state=42),
        "Boosting": GradientBoostingRegressor(random_state=42),
        "Linear": LinearRegression(),
    }

    # CV на масштабированном пространстве
    for name, model in models_t.items():
        y_cv_pred_scaled = cross_val_predict(model, X_scaled_t, y_scaled_t, cv=kf)
        
        # Обратное преобразование в исходную шкалу
        y_cv_pred = y_scaler_t.inverse_transform(y_cv_pred_scaled.reshape(-1, 1)).ravel()
        
        finite_mask = np.isfinite(y_cv_pred)
        rmse_cv_t = root_mean_squared_error(ytr_t[finite_mask], y_cv_pred[finite_mask])
        print(f"{name}: {rmse_cv_t:.2f}")

    # Обучение и тестирование
    Xte_t, yte_t, idx_te_t = _prepare_xy(test_df, FEATS, "T_delta_SPE")
    X_scaled_te_t = X_scaler_t.transform(Xte_t)

    for name, model in models_t.items():
        # Обучение на масштабированном пространстве
        model.fit(X_scaled_t, y_scaled_t)
        
        # Предсказание на тесте
        y_test_pred_scaled = model.predict(X_scaled_te_t)
        
        # Обратное преобразование в исходную шкалу
        y_test_pred = y_scaler_t.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
        
        finite_mask = np.isfinite(y_test_pred)
        rmse_test_t = root_mean_squared_error(yte_t[finite_mask], y_test_pred[finite_mask])
        print(f"Test SC25 {name} RMSE: {rmse_test_t:.2f}")
        new_table.loc[idx_te_t, f"{name}_Delta_T_max"] = y_test_pred

    # Важности для T_delta_SPE
    imp_t = build_importances_for_target(
        models_dict=models_t,
        X_train=X_scaled_t,
        y_train=y_scaled_t,
        feature_names=FEATS,
        is_jmax=False,
        y_scaler=None,
        inv_target=False
    )
    plot_separated_method_importances(
        imp_t, FEATS,
        title="T_delta_SPE — Builtin vs Permutation(ΔMSE) vs SHAP",
        output_path="plots/importance_tdelta.png",
        decimals=1
    )

    # 7) Сохранение
    if "Jmax_parsed" in df.columns and "T_delta_SPE" in df.columns:
        new_table["Jmax_parsed"] = pd.to_numeric(df["Jmax_parsed"], errors="coerce")
        new_table["T_delta_SPE"] = pd.to_numeric(df["T_delta_SPE"], errors="coerce")

    new_table.to_excel(OUTPUT_XLSX, index=False)
    print(f"\nРезультаты сохранены в '{OUTPUT_XLSX}'")
    print("Модели обучались на масштабированном пространстве (X и y)")


if __name__ == "__main__":
    main()
