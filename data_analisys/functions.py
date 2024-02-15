import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
from itertools import groupby


def preprocessing(file, sep, columns):
    data = pd.read_csv(file, sep=sep, header=None)
    data.columns = columns

    col = data[data.keys()[0]].tolist()
    for i in range(len(col)):
        data[data.keys()[0]] = data[data.keys()[0]] \
            .replace(col[i], datetime.strptime(col[i][:-7], "%Y-%m-%d %H:%M:%S"))

    for j in range(1, 3):
        col = data[data.keys()[j]].tolist()
        for i in range(len(col)):
            check = re.match(r" *[\d.]+ *", col[i])
            if check:
                data[data.keys()[j]] = data[data.keys()[j]].replace(col[i], float(col[i]))
            else:
                data[data.keys()[j]] = data[data.keys()[j]].replace(col[i], None)

    data = data.dropna()
    return data


def get_events(table: pd.DataFrame, column: str, value: float = 1.5):
    events = [list(g) for k, g in groupby(table[column].tolist(), key=lambda x: x > value) if k]

    return events


def create_graph(x_axis: list, y_axis: list):
    plt.plot(x_axis, y_axis)
    plt.semilogy()
    plt.show()
