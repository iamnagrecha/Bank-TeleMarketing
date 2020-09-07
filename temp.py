import pandas as pd
import numpy as np
from sklearn import model_selection

import missing_values
import feature_deletation
import categorical2numerical
import feature_selection
import negative2positive
import normalization
import imbalanced_problem
import classifiers

pd.set_option('display.max_columns', 55)

# reading the dataset
df = pd.read_csv('bank-additional-full.csv', sep=';')
df["y"] = df["y"].map({"no": 0, "yes": 1})

# feature deletation
feature_deletation.apply(df)

# dealing with missing values

UNKNOWN_FEATURES = ['job', 'marital', 'education', 'housing', 'loan', 'default', 'poutcome']
missing_values.apply_regression_imputation(df, "job")
# missing_values.apply_regression_imputation(df, "education")
# missing_values.apply_regression_imputation(df, "default")


