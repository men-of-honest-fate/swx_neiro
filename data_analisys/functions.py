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
    time = table["Date/time"].tolist()
    new_time = []
    tmp = []
    for event in events:
        for index in [event.index(value) for value in event]:
            tmp.append(time[index])
        new_time.append(tmp)
        tmp = []

    return (events, new_time)


def filter_events(events: tuple, indexes: list | None = None):
    if indexes:
        new_tuple = ([events[0][index] for index in indexes], [events[1][index] for index in indexes])
    else:
        new_tuple = ([event for event in events[0] if len(event) > 10], [events for event in events[1] if len(event) > 10])

    return new_tuple



def create_graph(x_axis: list, y_axis: list):
    plt.plot(x_axis, y_axis)
    plt.semilogy()
    plt.show()
