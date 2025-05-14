from data_analisys import functions as f
from data.parce_data import df
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import matplotlib.pyplot as plt
import warnings
import io


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    new_table = df[["Event_date", "T_delta_flare", "Flare_power"]]

    X_parced = np.array(df[["T_delta_flare", "Flare_power"]], dtype=np.float64)
    y_parced = np.array(df[["Jmax_parsed"]], dtype=np.float64)
    y_log = np.log1p(y_parced)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler_X_J = StandardScaler()
    scaler_y_J = StandardScaler()
    X = scaler_X_J.fit_transform(X_parced)
    y = scaler_y_J.fit_transform(y_log.reshape(-1, 1))

    models_J = {
        "Forest": RandomForestRegressor(random_state=42),
        "Boosting": GradientBoostingRegressor(random_state=42),
        "Linear": LinearRegression()
    }

    for name, model in models_J.items():
        model.fit(X, y)
        pred_scaled = model.predict(X)
        model_score = cross_val_score(model, X, y, cv=kf, scoring='r2')
        print(f"{model.__class__.__name__} CV Accuracy: {np.mean(model_score):.2f}")
        # Обратное преобразование предсказаний
        pred = np.expm1(scaler_y_J.inverse_transform(pred_scaled.reshape(-1, 1))).ravel()
        new_table[f"{name}_Jpred"] = pred

    new_table["True_J_values"] = y_parced
    column_order = [
        "Event_date", 
        "T_delta_flare", 
        "Flare_power", 
        "True_J_values",
        "Forest_Jpred", 
        "Boosting_Jpred", 
        "Linear_Jpred"
    ]
    new_table[column_order].to_excel("predictions_table_J.xlsx", index=False)

    X_parced = np.array(df[["T_delta_flare", "Flare_power"]], dtype=np.float64)
    y_parced = np.array(df[["T_delta_SPE"]], dtype=np.float64)
    scaler_X_T = StandardScaler()
    scaler_y_T = StandardScaler()
    X = scaler_X_T.fit_transform(X_parced)
    y = scaler_y_T.fit_transform(y_parced)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    models_T = {
        "Forest": RandomForestRegressor(random_state=42),
        "Boosting": GradientBoostingRegressor(random_state=42),
        "Linear": LinearRegression()
    }

    for name, model in models_T.items():
        model.fit(X, y)
        pred_scaled = model.predict(X)
        model_score = cross_val_score(model, X, y, cv=kf, scoring='r2')
        print(f"{model.__class__.__name__} CV Accuracy: {np.mean(model_score):.2f}")
        # Обратное преобразование предсказаний
        pred = scaler_y_T.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        new_table[f"{name}_Delta_T_max"] = pred


    new_table["True_Delta_T_max"] = y_parced
    column_order = [
        "Event_date", 
        "T_delta_flare", 
        "Flare_power", 
        "True_Delta_T_max",
        "Forest_Delta_T_max", 
        "Boosting_Delta_T_max", 
        "Linear_Delta_T_max"
    ]
    new_table[column_order].to_excel("predictions_table_T.xlsx", index=False)
