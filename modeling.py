import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from scipy import sparse

import seaborn as sns
sns.set(color_codes=True)

import matplotlib.pyplot as plt
import statsmodels.api as sm
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

helpers.draw(y=val_errors, xlabel="iteration", ylabel="MSE",
             filename='./imgs/mean_squared_error.png', title="Mean Squared Error")

results = sm.OLS(y_pred,sm.add_constant(y_test)).fit()
print(results.summary())

sns_plot = sns.lmplot(x="True Data", y='Predicted Data', data=pd.DataFrame(list(zip(y_test, y_pred)), columns =['True Data', 'Predicted Data']), fit_reg=True)
fig = sns_plot.fig
fig.suptitle('True VS Predicted log_prices', fontsize=8)
fig.savefig('./imgs/true_vs_predicted_log_prices.png')
plt.show()
# notice the confidence interval is very small; we almost can't see it
