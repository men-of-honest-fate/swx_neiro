import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
from itertools import groupby


def preprocessing(file, sep, columns):
    data = pd.read_csv(file, sep=sep, header=None)
    data.columns = columns

    col = data[data.keys()[0]].tolist()
    for i in range(len(col)):
        data[data.keys()[0]] = data[data.keys()[0]].replace(
            col[i], datetime.strptime(col[i][:-7], "%Y-%m-%d %H:%M:%S")
        )

    for j in range(1, 2):
        col = data[data.keys()[j]].tolist()
        for i in range(len(col)):
            check = re.match(r" *[\d.]+ *", col[i])
            if check:
                data[data.keys()[j]] = data[data.keys()[j]].replace(
                    col[i], float(col[i])
                )
            else:
                data[data.keys()[j]] = data[data.keys()[j]].replace(col[i], None)

    data = data.dropna()
    return data


def table_parce(file, sep):
    with open(file, "r") as f:
        columns = [column.replace("\n", "") for column in f.readline().split(sep)]
        data = [
            [obj.replace("\n", "") for obj in row.split(sep)] for row in f.readlines()
        ]
        df = pd.DataFrame(columns=columns, data=data)
        df = df.drop(columns=["T0 CME (Day/UT)", "CME (data)", "AR", "GLE", "Оі1", "Eqmmax1 (MeV)"])
        df = df.drop(df[df["Source max 1"] == "Unknown"].index)
        df = df.drop(df[df["Source max 1"] == ""].index)
        df = df.drop(df[df["Source max 1"] == "DSF"].index)
        df = df.drop(columns=["Source max 1", "Confidence of The Source Association"])
        df = df.drop(df[df["Importance (Xray/Opt)"] == ""].index)
        df = df.drop(df[df["Importance (Xray/Opt)"] == "26o"].index)
        df = df.drop(df[df["Localization"] == ""].index)
        df = df.dropna()
        loc_x = []
        loc_y = []
        delta_values = []
        for index, row in df.iterrows():
            # Time delta calculation
            row["Start (Day/UT)"] = row["Start (Day/UT)"][-3:-1]
            if "d" in row["Tmax1 (UT)"]:
                try:
                    row["Tmax1 (UT)"] = int(row["Tmax1 (UT)"][-3:-1])
                except ValueError:
                    row["Tmax1 (UT)"] = int(row["Tmax1 (UT)"][-2:-1])
                if row["Tmax1 (UT)"] - int(row["Start (Day/UT)"]) < 0:
                    delta_values.append(
                        24 + int(row["Tmax1 (UT)"]) - int(row["Start (Day/UT)"])
                    )
                else:
                    delta_values.append(
                        int(row["Tmax1 (UT)"]) - int(row["Start (Day/UT)"])
                    )
            else:
                row["Tmax1 (UT)"] = int(row["Tmax1 (UT)"][:-1])
                delta_values.append(int(row["Tmax1 (UT)"]) - int(row["Start (Day/UT)"]))

            # Importance of solar Flare calculation
            row["Importance (Xray/Opt)"] = (
                row["Importance (Xray/Opt)"].split("/")[0].split(",")[0]
            )
            try:
                ind = row["Importance (Xray/Opt)"][0]
                value = float(row["Importance (Xray/Opt)"][1:])
            except ValueError:
                ind = row["Importance (Xray/Opt)"][1:]
                value = float(row["Importance (Xray/Opt)"][0])
            if ind == "B":
                row["Importance (Xray/Opt)"] = np.float64(value * (10**-7))
            elif ind == "C":
                row["Importance (Xray/Opt)"] = np.float64(value * (10**-6))
            elif ind == "M":
                row["Importance (Xray/Opt)"] = np.float64(value * (10**-5))
            elif ind == "X":
                row["Importance (Xray/Opt)"] = np.float64(value * (10**-4))
            elif ind == "С":
                row["Importance (Xray/Opt)"] = np.float64(value * (10**-6))
            elif ind == "М":
                row["Importance (Xray/Opt)"] = np.float64(value * (10**-5))

            # Localization of solar flare calculation

            localization_x = row["Localization"][3:6]
            if localization_x[0] == "E":
                loc_x.append(float(localization_x[1:]))
            elif localization_x[0] == "W":
                loc_x.append(-float(localization_x[1:]))

            localization_y = row["Localization"][0:3]
            if localization_y[0] == "N":
                loc_y.append(float(localization_y[1:]))
            elif localization_y[0] == "S":
                loc_y.append(-float(localization_y[1:]))

        df.insert(6, "Localization_x", value=loc_x)
        df.insert(7, "Localization_y", value=loc_y)
        df.insert(0, "Tmax delta", value=delta_values)
        df = df.drop(columns=["Start (Day/UT)", "Tmax1 (UT)", "Localization"])

        return df


def get_events(table: pd.DataFrame, column: str, value: float = 1.5):
    events = [
        list(g)
        for k, g in groupby(table[column].tolist(), key=lambda x: x > value)
        if k
    ]
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
        new_tuple = (
            [events[0][index] for index in indexes],
            [events[1][index] for index in indexes],
        )
    else:
        new_tuple = (
            [event for event in events[0] if len(event) > 10],
            [events for event in events[1] if len(event) > 10],
        )

    return new_tuple


def create_graph(x_axis: list, y_axis: list):
    plt.plot(x_axis, y_axis)
    plt.semilogy()
    plt.show()
