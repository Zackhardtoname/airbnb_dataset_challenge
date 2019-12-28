from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import sparse
from sklearn import ensemble
import pickle
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scripts import helpers

X = pd.DataFrame(sparse.load_npz("../data/X.npz").toarray())
y = pd.read_csv("../data/y.csv", index_col=0, header=None)
var_names = pickle.load(open('../data/var_names.pkl', 'rb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for dataset, name in [(X_train, "X_train"), (X_test, "X_test"), (y_train, "y_train"), (y_test, "y_test")]:
        dataset.to_csv("../data/train_test_data/" + name + ".csv")

y_train = y_train.values.reshape(-1,)
y_test = y_test.values.reshape(-1,)

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(60, 60))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importance relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
ranked_features_asc = np.array(var_names)[sorted_idx]
plt.yticks(pos, ranked_features_asc)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('../imgs/deviance.png')
print("Check out the plot located at '../imgs/deviance.png'")

ranked_features_dsc = ranked_features_asc[::-1]
print("Features of most importance (descending):\n")
# rank the features in descending order
higher_cutoff_idx = np.argmax(feature_importance[sorted_idx]>=5)
lower_cutoff_idx = np.argmax(feature_importance[sorted_idx]>=1)

second_level_idx = (len(feature_importance) - higher_cutoff_idx + 1) * ["More Important (>=5)"] + (higher_cutoff_idx - lower_cutoff_idx) * ["Less Important (1<= && <5)"] + \
                     (lower_cutoff_idx - 1) * ["Not Important (<1)"]
pd.DataFrame(feature_importance[sorted_idx][::-1], index=[second_level_idx, ranked_features_dsc])

results = sm.OLS(y_pred,sm.add_constant(y_test)).fit()
print(results.summary())

sns_plot = sns.lmplot(x="True Data", y='Predicted Data', data=pd.DataFrame(list(zip(y_test, y_pred)), columns =['True Data', 'Predicted Data']), fit_reg=True)
fig = sns_plot.fig
fig.suptitle('True VS Predicted log_prices', fontsize=8)
fig.savefig('../imgs/true_vs_predicted_log_prices.png')
plt.show()

########################################
# Explanation of the results
#
#
#
########################################