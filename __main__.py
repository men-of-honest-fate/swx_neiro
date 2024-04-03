from data_analisys import functions as f
from settings import Settings
from ai_model import model

if __name__ == "__main__":
    model.neiro_fit()

    # model.neiro_fit()
    # columns = ["Date/time", "P >5 MeV", "P >10 MeV"]

    # settings = Settings().get_settings()
    # table = f.preprocessing(file=settings["path"],
    #                         sep=settings["sep"],
    #                         columns=columns)
    # events = f.get_events(table=table, column="P >5 MeV", value=0.5)
    # events = f.filter_events(events=events, indexes=[0,3,4])
    # for i in range(len(events[1])):
    #     f.create_graph(y_axis=events[0][i], x_axis=[i for i in range(len(events[0][i]))])
