import warnings
from data_analisys import functions as f
from settings import Settings
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import CubicSpline


def neiro_fit():
    warnings.filterwarnings('ignore')

    settings = Settings().get_settings()
    columns = ["Date/time", "P >30 MeV", "P >50 MeV"]
    table = f.preprocessing(file=settings["path"],
                            sep=settings["sep"],
                            columns=columns)
    events = f.get_events(table=table, column="P >30 MeV", value=0.5)
    events = f.filter_events(events=events)

    # fit_model = nn.Sequential(
    #     nn.Linear(1,5, bias=True),
    #     nn.LogSigmoid(),
    #     nn.Linear(5,300, bias=True),
    #     nn.LogSigmoid(),
    #     nn.Linear(300,1, bias=True),
    #     nn.LogSigmoid()
    # )
    for i in range(len(events[0])):
        X_test = torch.tensor([float(i) for i in range(len(events[0][i]))])
        Y_test = torch.tensor(events[0][i])
        spline = CubicSpline(X_test, Y_test)
        Y_pred = spline(X_test)

        plt.figure(figsize=(20,7))
        plt.scatter(x=X_test, y=Y_test, label='True Event')
        plt.plot(X_test, Y_pred, label="Prediction")
        plt.legend()
        plt.show()
