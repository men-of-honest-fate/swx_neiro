from data_analisys import functions as f
from ai_models import RandomForest, GradientBoosting, Linear
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import warnings
import io

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    table = f.table_parce(file="dump.txt", sep="\t")
    table_in = [
        [row["Localization_x"], row["Localization_y"], row["Importance (Xray/Opt)"]]
        for index, row in table.iterrows()
    ]
    table_out1 = [[row["Jmax1 (pfu)"]] for index, row in table.iterrows()]
    X_train, X_test, y_train, y_test = train_test_split(
        table_in, table_out1, test_size=0.1, random_state=16
    )
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)
    
    forest_model, Forest_accuracy = RandomForest.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    boosting_model, Boosting_accuracy = GradientBoosting.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    linear_model, Linear_accuracy = Linear.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    table.insert(6, "Forest_Jpred", forest_model.predict(np.array(table_in, dtype=np.float64)))
    table.insert(7, "Boosting_Jpred", boosting_model.predict(np.array(table_in, dtype=np.float64)))
    table.insert(8, "Linear_Jpred", linear_model.predict(np.array(table_in, dtype=np.float64)))
    
    def create_graphs_J():
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

        plt.legend(['RandomForest_J', 'GradientBoosting_J'], loc=4)
        plt.show()

    table_out2 = [[row["Tmax delta"]] for index, row in table.iterrows()]

    X_train, X_test, y_train, y_test = train_test_split(
        table_in, table_out2, test_size=0.1, random_state=16
    )
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)

    forest_model, Forest_accuracy = RandomForest.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    boosting_model, Boosting_accuracy = GradientBoosting.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    linear_model, Linear_accuracy = Linear.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    table.insert(9, "Forest_DTpred", forest_model.predict(np.array(table_in, dtype=np.float64)))
    table.insert(10, "Boosting_DTpred", boosting_model.predict(np.array(table_in, dtype=np.float64)))
    table.insert(11, "Linear_DTpred", linear_model.predict(np.array(table_in, dtype=np.float64)))

    print(Forest_accuracy, Boosting_accuracy)
    def create_graphs_DT():
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

        plt.legend(['RandomForest_DT', 'GradientBoosting_DT'], loc=4)
        plt.show()
    
    table.to_excel("dump.xlsx")
