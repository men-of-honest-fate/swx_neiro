from data_analisys.functions import preprocessing, get_events
from settings import Settings


if __name__ == "__main__":
    columns = ["Date/time", "P >5 MeV", "P >10 MeV"]

    settings = Settings().get_settings()
    table = preprocessing(file=settings["path"],
                          sep=settings["sep"],
                          columns=columns)
    events = get_events(table)
    print(events)
    print(len(events))
