from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample
from sklearn.metrics import r2_score


def fit(X_train, y_train, X_test, y_test):
    accuracy_array = []
    # n_bootstraps = 300
    # bootstrap_X = []
    # bootstrap_y = []

    # for _ in range(n_bootstraps):
    #     sample_X, sample_y = resample(X_train, y_train, random_state=16)
    #     bootstrap_X.append(sample_X)
    #     bootstrap_y.append(sample_y)

    # for i, data in enumerate(bootstrap_X):
    model = GradientBoostingRegressor(
        n_estimators=200, max_features="sqrt", random_state=42
    )
    model.fit(X_train, y_train)
    accuracy_score = r2_score(y_true=y_test, y_pred=model.predict(X_test))
    accuracy_array.append(accuracy_score)

    return model, max(accuracy_array)
