from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score


def fit(X_train, y_train, X_test, y_test):
    accuracy_array = []

    for i in range(y_train.shape[1]):
        for n in range(1, 201):
            model = GradientBoostingRegressor(
                n_estimators=n, max_features="sqrt", random_state=42
            )
            model.fit(X_train, y_train)
            accuracy_score = r2_score(y_test, model.predict(X_test))
            accuracy_array.append(accuracy_score)

    return model, accuracy_array
