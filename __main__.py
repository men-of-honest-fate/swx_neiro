from data_analisys import functions as f
from settings import Settings

if __name__ == "__main__":
    columns = ["Date/time", "P >5 MeV", "P >10 MeV"]

    settings = Settings().get_settings()
    table = f.preprocessing(file=settings["path"],
                            sep=settings["sep"],
                            columns=columns)
    # print(table)
    f.create_graph(x_axis=table[table.keys()[0]].tolist(), y_axis=table[table.keys()[2]].tolist())
    events = f.get_events(table=table, column="P >5 MeV", value=2.0)
    print(events)
