import warnings
from data_analisys import functions as f
from sklearn.metrics import accuracy_score
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def neiro_fit():
    warnings.filterwarnings("ignore")

    table = f.table_parce(file="dump.txt", sep="\t")
    table_in = [
        [row["Localization_x"], row["Localization_y"], row["Importance (Xray/Opt)"]]
        for index, row in table.iterrows()
    ]
    table_out1 = [[row["Jmax1 (pfu)"]] for index, row in table.iterrows()]

    X_train, X_test, y_train, y_test = train_test_split(
        table_in, table_out1, test_size=0.33, random_state=42
    )
    models1 = [
        RandomForestRegressor(n_estimators=100, max_features="sqrt"),  # случайный лес
        KNeighborsRegressor(n_neighbors=2),  # метод ближайших соседей
        LinearRegression(),  # логистическая регрессия
    ]
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_train, dtype=np.float64)

    TestModels = DataFrame()
    tmp = {}
    i = 0
    # для каждой модели из списка
    for model in models1:
        # получаем имя модели
        m = str(model)
        tmp["Model"] = m[: m.index("(")]
        # для каждого столбцам результирующего набора
        # for i in range(len(y_train)):
        # обучаем модель
        for i in range(y_train.shape[1]):
            # обучаем модель
            model.fit(X_train, y_train)
            # вычисляем коэффициент детерминации
            tmp["R2_Y1"] = r2_score(y_test, model.predict(X_test))
        i += 1
        # записываем данные и итоговый DataFrame
        TestModels = TestModels._append([tmp])
    # делаем индекс по названию модели
    TestModels.set_index("Model", inplace=True)

    fig, axes = plt.subplots(ncols=2)
    TestModels.R2_Y1.plot(ax=axes[0], kind="bar", title="Jmax")

    table_out2 = [[row["Tmax delta"]] for index, row in table.iterrows()]
    X_train, X_test, y_train, y_test = train_test_split(
        table_in, table_out2, test_size=0.33, random_state=42
    )
    models2 = [
        RandomForestRegressor(n_estimators=100, max_features="sqrt"),  # случайный лес
        KNeighborsRegressor(n_neighbors=6),  # метод ближайших соседей
        LinearRegression(),  # логистическая регрессия
    ]
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_train, dtype=np.float64)
    TestModels = DataFrame()
    tmp = {}
    i = 0
    # для каждой модели из списка
    for model in models2:
        # получаем имя модели
        m = str(model)
        tmp["Model"] = m[: m.index("(")]
        # для каждого столбцам результирующего набора
        # for i in range(len(y_train)):
        # обучаем модель
        for i in range(y_train.shape[1]):
            # обучаем модель
            model.fit(X_train, y_train)
            # вычисляем коэффициент детерминации
            tmp["R2_Y2"] = r2_score(y_test, model.predict(X_test))
        i += 1
        # записываем данные и итоговый DataFrame
        TestModels = TestModels._append([tmp])
    TestModels.set_index("Model", inplace=True)
    TestModels.R2_Y2.plot(ax=axes[1], kind="bar", color="green", title="Delta T")
    plt.show()
