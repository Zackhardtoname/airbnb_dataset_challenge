#!/usr/bin/env python
# coding: utf-8

# # Pipeline for Data Pre-processing

# In[3]:


import pickle

from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn_pandas import CategoricalImputer

import helpers

# In[4]:


df = pickle.load(open('../data/df.pkl', 'rb'))
num_attribs = pickle.load(open('../data/num_attribs.pkl', 'rb'))
cat_attribs = pickle.load(open('../data/cat_attribs.pkl', 'rb'))


# ## Pipeline for imputing and standardizing data
# 
# Since the steps are very standard in almost any regression problem, I won't write too much explanation here.

# In[5]:


num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('imputer', CategoricalImputer(strategy='constant', fill_value='nan')), # mode by default; only works with string values
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

df = df[num_attribs + cat_attribs]
cols_contain_nans = df.columns[df.isna().any()].tolist()
cat_cols_contain_nans = [col for col in cols_contain_nans if col in cat_attribs]
print("cat_cols_contain_nans: \n", cat_cols_contain_nans)


# In[6]:


# to make sure the dependent variable is not in the training data
try:
    num_attribs.remove("log_price")
except:
    pass

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
])


# In[7]:


X = full_pipeline.fit_transform(df[num_attribs + cat_attribs])
y = df["log_price"]


# ## Save the Data

# In[8]:


sparse.save_npz("../data/X.npz", X)
var_names = helpers.get_transformer_feature_names(full_pipeline)
pickle.dump(var_names, open('../data/var_names.pkl', 'wb'))
y.to_csv("../data/y.csv")
