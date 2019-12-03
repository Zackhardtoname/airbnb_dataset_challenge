import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingRegressor
from scipy import sparse
from sklearn.metrics import r2_score
import helpers

X = pd.DataFrame(sparse.load_npz("./data/X.npz").toarray())
y = pd.read_csv("./data/y.csv", index_col=0, header=None)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for dataset, name in [(X_train, "X_train"), (X_test, "X_test"), (y_train, "y_train"), (y_test, "y_test")]:
        dataset.to_csv("./data/train_test_data/" + name + ".csv")

y_train = y_train.values.reshape(-1,)
y_test = y_test.values.reshape(-1,)

gbrt = GradientBoostingRegressor(max_depth=10, warm_start=True)
min_val_error = float("inf")
error_going_up = 0
val_errors = []

for n_estimators in range(1, 1200):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_test)
    val_error = mean_squared_error(y_test, y_pred)
    val_errors.append(val_error)
    print(val_error)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
    if error_going_up == 5:
        break # early stopping

helpers.draw(val_errors, './imgs/mean_squared_error_figure.png')

diff = y_pred - y_test
helpers.draw(diff[:20], "errors")