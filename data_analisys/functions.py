import pandas as pd


def preprocessing(file, sep, columns):
    data = pd.read_csv(file, sep=sep, header=None)
    data.columns = columns
    data.dropna(thresh=1)

    return data


def get_events(table: pd.DataFrame):
    events = []
    tmp = []
    is_event = False

    for i in range(1, table.shape[0]):
        try:
            val = float(table.iloc[i][table.keys()[2]])
        except ValueError:
            continue
        if val > 0.4:
            is_event = True
            tmp.append(val)
        else:
            tmp.append(0.0)
            if is_event:
                events.append(tmp)
            is_event = False

    return events
