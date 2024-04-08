from data_analisys import functions as f
from ai_models import RandomForest, Linear, GradientBoosting
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    table = f.table_parce(file="dump.txt", sep="\t")
    table_in = [
        [row["Localization_x"], row["Localization_y"], row["Importance (Xray/Opt)"]]
        for index, row in table.iterrows()
    ]
    table_out1 = [[row["Jmax1 (pfu)"]] for index, row in table.iterrows()]

    X_train, X_test, y_train, y_test = train_test_split(
        table_in, table_out1, test_size=0.1, random_state=42
    )
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_train, dtype=np.float64)

    Forest_accuracy = RandomForest.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)[1]
    # print("Linear accuracy:", Linear.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)[1])
    Boosting_accuracy = GradientBoosting.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)[1]
    ns = np.arange(1, 201, 1)
    plt.plot(ns, Forest_accuracy, ns, Boosting_accuracy, linewidth=2.0)
    plt.title('Models accuracy', fontsize=14, fontname='Times New Roman')
    plt.xlabel('N estimators', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Accuracy', fontsize=14, fontname='Times New Roman')
    plt.grid(True)

    Forest_accuracy_max = max(Forest_accuracy)
    xpos = Forest_accuracy.index(Forest_accuracy_max)
    xmax = ns[xpos]
    plt.annotate(f"{Forest_accuracy_max.__round__(4), xmax}", xy=(xmax, Forest_accuracy_max), xytext=(xmax-10, Forest_accuracy_max + 0.01))

    Boosting_accuracy_max = max(Boosting_accuracy)
    xpos = Boosting_accuracy.index(Boosting_accuracy_max)
    xmax = ns[xpos]
    plt.annotate(f"{Boosting_accuracy_max.__round__(4), xmax}", xy=(xmax, Boosting_accuracy_max), xytext=(xmax-10, Boosting_accuracy_max - 0.05))

    plt.legend(['RandomForest', 'GradientBoosting'], loc=4)
    plt.show()
