from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import r2_score


def fit(X_train, y_train, X_test, y_test):
    n_bootstraps = 1000
    bootstrap_X = []
    bootstrap_y = []

    for _ in range(n_bootstraps):
        sample_X, sample_y = resample(X_train, y_train)
        bootstrap_X.append(sample_X)
        bootstrap_y.append(sample_y)

    model = LinearRegression()

    for i, data in enumerate(bootstrap_X):
        model.fit(data, bootstrap_y[i])
        accuracy_score = r2_score(bootstrap_y[i], model.predict(data))

    return model, accuracy_score
