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


INPUT_CSV = "processed_sps_data.csv"
OUTPUT_XLSX = "predictions_results_oos_sc25.xlsx"


# ==============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def _prepare_xy(df: pd.DataFrame, feature_cols, target_col: str):
    """Формирует X, y и индекс строк после фильтрации NaN/inf."""
    feature_cols = list(feature_cols)
    cols = feature_cols + [target_col]
    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    mask = work[cols].apply(np.isfinite).all(axis=1)
    work = work[mask]
    return work[feature_cols].to_numpy(), work[target_col].to_numpy(), work.index


def permutation_importance_mse_custom(estimator, X, y, scorer, n_repeats=20, random_state=42):
    """
    Общая обёртка для permutation importance через переданный scorer.
    Возвращает сырые важности (Δошибка) длины n_features.
    """
    rng = np.random.default_rng(random_state)
    base_score = scorer(estimator, X, y)  # например, -MSE в исходной шкале
    importances = np.zeros(X.shape[1], dtype=float)

    for j in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            scores.append(scorer(estimator, Xp, y))
        delta = base_score - np.mean(scores)   # для -MSE это >= 0 при ухудшении
        importances[j] = max(delta, 0.0)

    return importances


def build_importances_for_target(
    models_dict: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list | tuple,
    *,
    # для Jmax, если модель обучалась на y_scaled (scaled логарифм)
    is_jmax: bool = False,
    y_scaler=None,   # StandardScaler обученный на y_log
    inv_target: bool = False,
):
    """
    Считает важности тремя методами для набора моделей и одной цели.
    Возвращает словарь {(model_name, method): importance_vector_%}.
    """
    feature_names = list(feature_names)
    out = {}

    # 1) Builtin (деревья) / |coef| (линейная) → нормируем до 100%
    for name, model in models_dict.items():
        fi = None
        if hasattr(model, "feature_importances_"):
            fi = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "named_steps"):
            for _, st in model.named_steps.items():
                if hasattr(st, "feature_importances_"):
                    fi = np.asarray(st.feature_importances_, dtype=float)
                    break
        # Fallback для линейной: модуль коэффициентов последнего шага
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

    # 2) Permutation (ΔMSE ≥ 0) → нормируем до 100%
    for name, model in models_dict.items():
        if is_jmax and inv_target and y_scaler is not None:
            # scorer для Jmax в исходной шкале
            def scorer(est, X, y_true):
                y_pred_scaled = est.predict(X)
                y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                y_pred = np.expm1(y_pred_log)
                return -mean_squared_error(y_true, y_pred)

            pimps = permutation_importance_mse_custom(
                model, X_train, y_train, scorer, n_repeats=50, random_state=42
            )
        else:
            # Для T_delta_SPE можно оставить встроенный путь
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

    # 3) SHAP (TreeExplainer/LinearExplainer), затем нормировка до 100%
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
    """
    Три подграфика (builtin, permutation, shap).
    Группировка ПО МОДЕЛЯМ: для каждой модели рядом стоят столбцы по всем признакам.

    importances_dict: {(model_name, method_name): vector_%}
    feature_names: список имён признаков (в порядке столбцов в векторе)
    """
    feature_names = list(feature_names)
    methods = ["builtin", "permutation", "shap"]
    models = sorted({k[0] for k in importances_dict.keys()})
    n_feats = len(feature_names)
    n_models = len(models)

    # Матрицы важностей по методам: (n_models, n_feats)
    mats = {}
    for method in methods:
        mat = np.zeros((n_models, n_feats), dtype=float)
        for mi, m in enumerate(models):
            vec = importances_dict.get((m, method), np.zeros(n_feats, dtype=float))
            mat[mi, :] = vec
        mats[method] = mat

    # Общий максимум и запас по оси Y для всех подграфиков,
    # чтобы подписи не вылезали за пределы.[web:3]
    global_max = max(mat.max() for mat in mats.values()) if mats else 0.0
    y_pad = global_max * 0.08  # 8% от высоты – запас под подписи

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

            # Подписи над столбиками с небольшим отступом,
            # clip_on=True гарантирует, что текст обрежется по границе осей при необходимости.[web:22][web:28]
            for rect in b:
                h = rect.get_height()
                if h > 0:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        h,
                        f"{h:.{decimals}f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        clip_on=True,
                    )

        ax.set_xticks(x_groups)
        ax.set_xticklabels(models)
        ax.set_ylabel("Вклад, %")
        ax.set_title(f"{method}", loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # Общие пределы Y для всех подграфиков
        ax.set_ylim(0, global_max + y_pad)

    fig.suptitle(title, fontsize=13)

    # Одна общая легенда по признакам на уровне фигуры.[web:7][web:9][web:11]
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.9, 0.9),
        ncol=min(len(feature_names), 4),
        title="Признаки",
        fontsize=9,
    )

    # Чуть ужать область под графики, чтобы у легенды и заголовка было место
    plt.tight_layout(rect=[0.0, 0.0, 0.9, 0.9])

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"График (3 подграфика; группировка по моделям) сохранён в '{output_path}'")
    plt.show()




# ==============================
# ОСНОВНОЙ СКРИПТ
# ==============================

def main():
    warnings.filterwarnings("ignore")

    # 1) Загрузка данных
    header = pd.read_csv(INPUT_CSV, nrows=0)
    parse_dates = [c for c in ["Event_date", "Tmax_parsed"] if c in header.columns]
    df = pd.read_csv(INPUT_CSV, parse_dates=parse_dates)

    # 2) Проверка схемы
    # Примечание: если в вашем CSV колонка называется "cycle", замените "Cycle" ниже на "cycle"
    required = {"T_delta_flare", "Flare_power", "Jmax_parsed", "T_delta_SPE", "Cycle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют требуемые колонки: {sorted(missing)}")

    # 3) Разделение по циклам (cycle только для split)
    cycle_num = pd.to_numeric(df["Cycle"], errors="coerce")
    df = df.assign(cycle_num=cycle_num)

    train_df = df[df["cycle_num"].isin([23, 24])].copy()
    test_df  = df[df["cycle_num"].isin([25])].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Недостаточно данных в train (cycles 23–24) или test (cycle 25).")

    # 4) Признаки
    FEATS = ["T_delta_flare", "Flare_power"]

    # 5) Модели для Jmax
    models_j = {
        "Forest": make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42)),
        "Boosting": make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42)),
        # "Linear": make_pipeline(StandardScaler(), LinearRegression()),
    }

    # 6) Модели для T_delta_SPE
    models_t = {
        "Forest": make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42)),
        "Boosting": make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42)),
        # "Linear": make_pipeline(StandardScaler(), LinearRegression()),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Таблица результатов
    base_cols = [c for c in ["Event_date", "T_delta_flare", "Flare_power", "Cycle"] if c in df.columns]
    new_table = df[base_cols].copy()
    new_table["split"] = np.where(df.index.isin(test_df.index), "test_sc25", "train_sc23_24")

    # =========================
    # Блок A: Jmax_parsed
    # =========================
    Xtr_j, ytr_j, idx_tr_j = _prepare_xy(train_df, FEATS, "Jmax_parsed")
    # Лог-скейл → стандартизация (как у вас)
    ytr_log = np.log1p(ytr_j)
    ysc = StandardScaler().fit(ytr_log.reshape(-1, 1))
    ytr_scaled = ysc.transform(ytr_log.reshape(-1, 1)).ravel()

    print("\n" + "=" * 40)
    print("Jmax_parsed — Train (cycles 23–24) CV RMSE")
    print("=" * 40)

    for name in models_j:
        new_table[f"{name}_Jpred"] = np.nan

    for name, model in models_j.items():
        ycv_scaled = cross_val_predict(model, Xtr_j, ytr_scaled, cv=kf)
        ycv_log = ysc.inverse_transform(ycv_scaled.reshape(-1, 1)).ravel()
        ycv = np.expm1(ycv_log)
        ycv = np.clip(ycv, 0.0, None)
        rmse_cv = root_mean_squared_error(ytr_j, ycv)
        print(f"{name}: {rmse_cv:.2f}")

    Xte_j, yte_j, idx_te_j = _prepare_xy(test_df, FEATS, "Jmax_parsed")
    for name, model in models_j.items():
        model.fit(Xtr_j, ytr_scaled)
        ypred_scaled = model.predict(Xte_j)
        ypred_log = ysc.inverse_transform(np.asarray(ypred_scaled).reshape(-1, 1)).ravel()
        ypred = np.expm1(ypred_log)
        ypred = np.clip(ypred, 0.0, None)
        rmse_test = root_mean_squared_error(yte_j, ypred)
        print(f"Test SC25 {name} RMSE: {rmse_test:.2f}")
        new_table.loc[idx_te_j, f"{name}_Jpred"] = ypred

    # Важности для Jmax
    imp_j = build_importances_for_target(
        models_dict=models_j,
        X_train=Xtr_j,
        y_train=ytr_j,    # важен таргет в исходной шкале для ΔMSE
        feature_names=FEATS,
        is_jmax=True,
        y_scaler=ysc,
        inv_target=True
    )
    plot_separated_method_importances(
        imp_j, FEATS,
        title="Вклад переменных, максимальная интенсивность",
        output_path="SHAP интенсивность.png",
        decimals=1
    )

    # =========================
    # Блок B: T_delta_SPE
    # =========================
    Xtr_t, ytr_t, idx_tr_t = _prepare_xy(train_df, FEATS, "T_delta_SPE")

    print("\n" + "=" * 40)
    print("Delta_T_max (T_delta_SPE) — Train (cycles 23–24) CV RMSE")
    print("=" * 40)

    for name in models_t:
        col = f"{name}_Delta_T_max"
        if col not in new_table.columns:
            new_table[col] = np.nan

    for name, model in models_t.items():
        ycv_t = cross_val_predict(model, Xtr_t, ytr_t, cv=kf)
        finite_mask = np.isfinite(ycv_t)
        rmse_cv_t = root_mean_squared_error(ytr_t[finite_mask], ycv_t[finite_mask])
        print(f"{name}: {rmse_cv_t:.2f}")

    Xte_t, yte_t, idx_te_t = _prepare_xy(test_df, FEATS, "T_delta_SPE")
    for name, model in models_t.items():
        model.fit(Xtr_t, ytr_t)
        ypred_t = model.predict(Xte_t)
        finite_mask = np.isfinite(ypred_t)
        rmse_test_t = root_mean_squared_error(yte_t[finite_mask], ypred_t[finite_mask])
        print(f"Test SC25 {name} RMSE: {rmse_test_t:.2f}")
        new_table.loc[idx_te_t, f"{name}_Delta_T_max"] = ypred_t

    # Важности для T_delta_SPE
    imp_t = build_importances_for_target(
        models_dict=models_t,
        X_train=Xtr_t,
        y_train=ytr_t,
        feature_names=FEATS,
        is_jmax=False,
        y_scaler=None,
        inv_target=False
    )
    # Для T_delta_SPE
    plot_separated_method_importances(
        imp_t, FEATS,
        title="Вклад переменных, фаза нарастания",
        output_path="SHAP фаза нарастания.png",
        decimals=1
    )

    # 7) Сохранение
    if "Jmax_parsed" in df.columns and "T_delta_SPE" in df.columns:
        new_table["Jmax_parsed"] = pd.to_numeric(df["Jmax_parsed"], errors="coerce")
        new_table["T_delta_SPE"] = pd.to_numeric(df["T_delta_SPE"], errors="coerce")
    else:
        # На случай, если вы фильтровали/переименовывали — можно подтянуть из train/test датафреймов
        if "Jmax_parsed" in train_df.columns:
            new_table["Jmax_parsed"] = pd.to_numeric(df["Jmax_parsed"], errors="coerce")
        if "T_delta_SPE" in train_df.columns:
            new_table["T_delta_SPE"] = pd.to_numeric(df["T_delta_SPE"], errors="coerce")

    new_table.to_excel(OUTPUT_XLSX, index=False)
    print(f"\nРезультаты сохранены в '{OUTPUT_XLSX}'")
    print("Столбцы 'Jmax_parsed' и 'T_delta_SPE' добавлены как истинные значения целей.")


if __name__ == "__main__":
    main()
