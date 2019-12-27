## Pipeline for imputing and standardizing data

import pickle
from data import data_keys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn_pandas import CategoricalImputer
from scipy import sparse

df = pickle.load(open('../data/df.pkl', 'rb'))

num_attribs = pickle.load(open('../data/num_attribs.pkl', 'rb'))
cat_attribs = pickle.load(open('../data/cat_attribs.pkl', 'rb'))

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
cat_cols_contain_nans

#%%

try:
    num_attribs.remove("log_price")
except:
    pass

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
])

X = full_pipeline.fit_transform(df[num_attribs + cat_attribs])
y = df["log_price"]

## Save the Data

sparse.save_npz("../data/X.npz", X)
y.to_csv("../data/y.csv")