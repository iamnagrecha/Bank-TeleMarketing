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
missing_values.apply_common_imputation(df)
missing_values.apply_mean_imputation(df)

# categorical 2 numerical attributes
# using label encoding and one-hot encoding
categorical2numerical.apply_label_encoding(df)
df = categorical2numerical.apply_onehot_encoding(df)

# negative to positive converter
negative2positive.apply(df)

# feature selection
feature_selection.apply_univariate_selection(df)
# feature_selection.show_correlation_matrix(df)
feature_selection.apply_correlation_filter_method(df)

# normalization
# normalization.apply_min_max(df)
# normalization.apply_standard_scalar(df)

# deal with imbalanced dataset
# over-sampling method (SMOTE)
# x, y = imbalanced_problem.apply_smote(df)
x, y = imbalanced_problem.apply_near_miss(df)

# train / test data split
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)

lr_hyperparameters = classifiers.get_lr_hyper_parameters(x_train, y_train)
# classifiers.apply_lr(x_train, y_train, x_test, y_test)
classifiers.apply_lr(x_train, y_train, x_test, y_test, lr_hyperparameters)

# dt_hyperparameters = classifiers.get_dt_hyper_parameters(x_train, y_train)
# classifiers.apply_dt(x_train, y_train, x_test, y_test)
# classifiers.apply_dt(x_train, y_train, x_test, y_test, dt_hyperparameters)

# rf_hyperparameters = classifiers.get_rf_hyper_parameters(x_train, y_train)
# classifiers.apply_rf(x_train, y_train, x_test, y_test)
# classifiers.apply_rf(x_train, y_train, x_test, y_test, rf_hyperparameters)

# mlp_hyperparameters = classifiers.get_mlp_hyper_parameters(x_train, y_train)
# classifiers.apply_mlp(x_train, y_train, x_test, y_test)
# classifiers.apply_mlp(x_train, y_train, x_test, y_test, mlp_hyperparameters)

# svm_hyperparameters = classifiers.get_svm_hyper_parameters(x_train, y_train)
# classifiers.apply_svm(x_train, y_train, x_test, y_test)
# classifiers.apply_svm(x_train, y_train, x_test, y_test, svm_hyperparameters)

# knn_hyperparameters = classifiers.get_knn_hyper_parameters(x_train, y_train)
# classifiers.apply_knn(x_train, y_train, x_test, y_test)
# classifiers.apply_knn(x_train, y_train, x_test, y_test, knn_hyperparameters)

# print(df[:100])


