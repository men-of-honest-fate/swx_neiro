from data_analisys import functions as f
from data.parce_data import df
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
import numpy as np
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Инициализация DataFrame
    new_table = df[["Event_date", "T_delta_flare", "Flare_power"]].copy()

    # Конвейер обработки для Jmax_parsed
    def process_jmax():
        X = df[["T_delta_flare", "Flare_power"]].values
        y = df["Jmax_parsed"].values
        
        # Логарифмическое преобразование
        y_log = np.log1p(y)
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_log.reshape(-1, 1)).ravel()
        
        # Модели с пайплайнами
        models = {
            "Forest": make_pipeline(
                StandardScaler(),
                RandomForestRegressor(random_state=42)
            ),
            "Boosting": make_pipeline(
                StandardScaler(),
                GradientBoostingRegressor(random_state=42)
            ),
            "Linear": make_pipeline(
                StandardScaler(),
                LinearRegression()
            )
        }

        print("\n" + "="*40)
        print("Оценка моделей для Jmax_parsed (RMSE)")
        print("="*40)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            y_pred_log = cross_val_predict(model, X, y_scaled, cv=kf)
            y_pred = np.expm1(scaler_y.inverse_transform(y_pred_log.reshape(-1, 1))).ravel()
            
            rmse = root_mean_squared_error(y, y_pred)
            print(f"{name}: {rmse:.2f}")
            new_table[f"{name}_Jpred"] = y_pred

    # Конвейер обработки для Delta_T_max
    def process_delta_t():
        X = df[["T_delta_flare", "Flare_power"]].values
        y = df["T_delta_SPE"].values
        
        models = {
            "Forest": make_pipeline(
                StandardScaler(),
                RandomForestRegressor(random_state=42)
            ),
            "Boosting": make_pipeline(
                StandardScaler(),
                GradientBoostingRegressor(random_state=42)
            ),
            "Linear": make_pipeline(
                StandardScaler(),
                LinearRegression()
            )
        }

        print("\n" + "="*40)
        print("Оценка моделей для Delta_T_max (RMSE)")
        print("="*40)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            y_pred = cross_val_predict(model, X, y, cv=kf)
            rmse = root_mean_squared_error(y, y_pred)
            print(f"{name}: {rmse:.2f}")
            new_table[f"{name}_Delta_T_max"] = y_pred

    # Запуск обработки
    process_jmax()
    process_delta_t()

    # Сохранение результатов
    new_table.to_excel("predictions_results.xlsx", index=False)
