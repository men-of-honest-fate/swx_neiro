from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def fit(X_train, y_train, X_test, y_test):
    accuracy_array = []

    for i in range(y_train.shape[1]):
        model = RandomForestRegressor(
            n_estimators=200, max_features="sqrt", random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test) + model.predict(X_train)
        print(model.predict(X_test), "\n", model.predict(X_train))
        accuracy_score = r2_score(y_true=y_test, y_pred=model.predict(X_test))
        accuracy_array.append(accuracy_score)

    return model, accuracy_array, predictions
