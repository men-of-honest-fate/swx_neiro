from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.utils import resample


def fit(X_train, y_train, X_test, y_test):
    n_bootstraps = 100
    bootstrap_X = []
    bootstrap_y = []

    for _ in range(n_bootstraps):
        sample_X, sample_y = resample(X_train, y_train, random_state=16)
        bootstrap_X.append(sample_X)
        bootstrap_y.append(sample_y)

    accuracy_array = []
    for i, data in enumerate(bootstrap_X):
        model = RandomForestRegressor(
            n_estimators=200, max_features="sqrt", random_state=42
        )
        model.fit(data, bootstrap_y[i])
        accuracy_score = r2_score(y_test, model.predict(X_test))
        accuracy_array.append(accuracy_score)

    return model, max(accuracy_array)
