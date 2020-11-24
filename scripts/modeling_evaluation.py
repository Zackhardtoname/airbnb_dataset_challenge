#!/usr/bin/env python
# coding: utf-8

# # Modeling
# 
# I chose Gradient boost regression tree (GBRT) for its high level of interpretability and relative ease of training.
# 

# In[98]:


import pickle
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from IPython.display import HTML
from scipy import sparse
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, ShuffleSplit,
                                     learning_curve, train_test_split)

import helpers

sns.set(color_codes=True)


# ## Grid Search

# In[99]:


# runtime parameter
grid_search = False


# In[100]:


# Part of the rationale behind the train test split is inspired by this post: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

X = pd.DataFrame(sparse.load_npz("../data/X.npz").toarray())
y = pd.read_csv("../data/y.csv", index_col=0, header=None)
var_names = pickle.load(open('../data/var_names.pkl', 'rb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

for dataset, name in [(X_train, "X_train"), (X_test, "X_test"), (y_train, "y_train"), (y_test, "y_test")]:
        dataset.to_csv("../data/train_test_data/" + name + ".csv")

y_train = y_train.values.reshape(-1,)
y_test = y_test.values.reshape(-1,)

params = {'n_estimators': [500], 'max_depth': range(4, 15, 4), 'min_samples_split': range(2, 1003, 400), 
          'min_samples_leaf':range(30,71,20), 'max_features': ["sqrt", "log2", None], 'subsample':[0.7,0.75,0.8,0.85],
            }

rgs = ensemble.GradientBoostingRegressor(learning_rate=0.01, loss='ls', warm_start=True)


# Warning: the following cell could take a while (around 2 hours in my case) to run
# I could search for the best parameters in groups, but here I chose to search all together to keep the code short and straightforward. 

# In[101]:


if grid_search:
    g_search = GridSearchCV(estimator=rgs, param_grid=params, scoring='neg_mean_squared_error', n_jobs=-2, iid=False, cv=3, verbose=10)
    g_search.fit(X_train, y_train)
    joblib.dump(g_search, open('../models/g_search.pkl', 'wb'))
else:
    g_search = joblib.load(open('../models/g_search.pkl', 'rb'))


# In[102]:


print(g_search.best_params_)
print(g_search.best_score_)


# In[103]:


rgs = g_search.best_estimator_
rgs.fit(X_train, y_train)


# In[104]:


mse = mean_squared_error(y_test, rgs.predict(X_test))
print("Final MSE: %.4f" % mse)


# In[105]:


# Plot training deviance

n_estimators = g_search.best_params_['n_estimators']
# compute test set deviance
test_score = np.zeros((n_estimators,), dtype=np.float64)

for i, y_pred in enumerate(rgs.staged_predict(X_test)):
    test_score[i] = rgs.loss_(y_test, y_pred)

plt.figure(figsize=(60, 60))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(n_estimators) + 1, rgs.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(n_estimators) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# Plot feature importance

feature_importance = rgs.feature_importances_
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


# Check out [the saved plot](../imgs/deviance.png) to see the actual texts.
# 
# We can see that as we increase the number of estimators for the tree, the errors decrease. 
# Thus, we would increase the number of estimators for better performance.

# We can also take a look at the learning curve to examine the bias and variance of our model.
# 
# Warning: the cell below could take a while

# In[106]:



# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
params = g_search.best_params_
params["learning_rate"] = .01
params["loss"] = 'ls'
params["warm_start"] = True

train_scores_mean = helpers.plot_learning_curve(ensemble.GradientBoostingRegressor(**params), "Learning Curve", X_train, y_train, n_jobs=-1) 
plt.show()


# For the learning curve graph, there is a large gap between the curves for the training and cross-validation scores. 
# That means there is a high level of variance, and getting more data and/or more training would help lower the error further.
# 
# For the scalability graph, the fit time goes up linearly with the number of training examples, indicating high scalability of the GBRT model.
# 
# For the performance of the model, we do see a reasonable diminishing rate of return, as in many other types of models, it is what I would expect.    

# Now we can further classify the variables by their importance.

# In[107]:


ranked_features_dsc = ranked_features_asc[::-1]
print("Features of most importance (descending):\n")
# rank the features in descending order
higher_cutoff_idx = np.argmax(feature_importance[sorted_idx]>=5)
lower_cutoff_idx = np.argmax(feature_importance[sorted_idx]>=1)

second_level_idx = (len(feature_importance) - higher_cutoff_idx + 1) * ["More Important (>=5)"] + (higher_cutoff_idx - lower_cutoff_idx) * ["Less Important (1<= && <5)"] +                      (lower_cutoff_idx - 1) * ["Not Important (<1)"]
feature_df = pd.DataFrame(feature_importance[sorted_idx][::-1], index=[second_level_idx, ranked_features_dsc], columns=["Importance"])

HTML(feature_df.to_html())


# # Conclusion on variables
# 
# Most of the numerical variables mentioned in the hypothesis are classified as "more importance" or "less important." 
# Note that many transformed and engineered numerical variables such as "log_bedrooms_per_accommodates" turn out to be of "more importance." 
# 
# On the other hand, only a few categorical variables tend out to be "more important" or "less important." 
# 
# Variables that are important but were not in the hypothesis are "response_time" and "property_type." 
# 
# We could take a closer look at the variables that we missed out by looking back at their distributions. We would exam "minimum_nights" and "maximum_nights" here as an example.
# 
# Overall, the part about important variables in the hypothesis is mainly correct for this model.

# In[108]:


df = pd.read_csv("../data/listings.csv", converters={"host_verifications": literal_eval})
df["minimum_nights"].describe()


# In[109]:


results = sm.OLS(y_pred,sm.add_constant(y_test)).fit()
print(results.summary())


# # Conclusion on R-squared
# 
# The R-squared value is .565, meaning the variables used in training could explain 56.5% of the variation in the log price. 
# P values for both the intercept and the slope for are 0, meaning they are statistically significant. 
# It is slightly below the value in our hypothesis, but still relatively satisfactory. 
# 
# For the fitted line, the slope (x1 in the table) is .565, which is not great since ideally, we would like a slope of 1. Yet .565 is as expected for us 
# since it should be close to the R-squared value (for some intuition, read this [post](https://stats.stackexchange.com/questions/87963/does-the-slope-of-a-regression-between-observed-and-predicted-values-always-equa)).
# The intercept (called "constant" in the table) is 2, suggesting when the real price approaches 0, the predicted price would almost be four times as much as the real price 
# (of course, that is an extremely unlikely or unrealistic outlier). In part, the poor intercept value results from the poor slope value. 
#    

# We can also draw a scatter plot of our predictions against the true prices in the test dataset and draw a fit.

# In[110]:


sns_plot = sns.lmplot(x="True Data", y='Predicted Data', data=pd.DataFrame(list(zip(y_test, y_pred)), columns =['True Data', 'Predicted Data']), fit_reg=True)
fig = sns_plot.fig
fig.suptitle('True VS Predicted log_prices', fontsize=8)
fig.savefig('../imgs/true_vs_predicted_log_prices.png')
plt.show()


# The plot looks great with residual relatively evenly distributed around the fitted line, with a very limited number of outliers. 

# # Future Improvement
# 1. I could use auto-transformers (https://datamadness.github.io/Skewness_Auto_Transform) for feature transformation as log transformation is not the only choice.
# 
# 2. Some variables are not used in this challenge. Most notably, the text-based variables such as "description," "neighborhood_overview," and "transit" would likely have a significant amount of impact on the prices.
# A direction for future improvement could be to convert them into numerical variables by taking the average pooling of the word vectors.
# 
# 3. I could also try different models other than GBRT.
# 
# 4. I could use bayesian optimization instead of grid search to save time in finding the best parameters.
# 
