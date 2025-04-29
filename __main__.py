from data_analisys import functions as f
from data.parce_data import df
from ai_models import RandomForest, GradientBoosting, Linear
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import warnings
import io


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # table = f.table_parce(file="dump.txt", sep="\t")
    # print(table.info())
    # table_in = [
    #     [row["Localization_x"], row["Localization_y"], row["Importance (Xray/Opt)"]]
    #     for index, row in table.iterrows()
    # ]
    # table_out1 = [[row["Jmax1 (pfu)"]] for index, row in table.iterrows()]
    # X_train, X_test, y_train, y_test = train_test_split(
    #     table_in, table_out1, test_size=0.1, random_state=16
    # )

    print(df.info())
    X = df[["T_delta_flare", "Flare_power"]]
    y = df[["Jmax_parsed"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)
    
    forest_model, Forest_accuracy = RandomForest.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    boosting_model, Boosting_accuracy = GradientBoosting.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    linear_model, Linear_accuracy = Linear.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    # table.insert(6, "Forest_Jpred", forest_model.predict(np.array(table_in, dtype=np.float64)))
    # table.insert(7, "Boosting_Jpred", boosting_model.predict(np.array(table_in, dtype=np.float64)))
    # table.insert(8, "Linear_Jpred", linear_model.predict(np.array(table_in, dtype=np.float64)))
    print(Forest_accuracy, Boosting_accuracy)
    
    # table_out2 = [[row["Tmax delta"]] for index, row in table.iterrows()]

    # X = df[["T_delta_flare", "Flare_power"]]
    # Y = df[["T_delta_SPE"]]
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, Y, test_size=0.1, random_state=42
    # )
    # X_train = np.array(X_train, dtype=np.float64)
    # X_test = np.array(X_test, dtype=np.float64)
    # y_train = np.array(y_train, dtype=np.float64)
    # y_test = np.array(y_test, dtype=np.float64)

    # forest_model, Forest_accuracy = RandomForest.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    # boosting_model, Boosting_accuracy = GradientBoosting.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    # linear_model, Linear_accuracy = Linear.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # print(Forest_accuracy, Boosting_accuracy)
