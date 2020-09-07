
import copy
import pandas as pd
from sklearn.linear_model import LogisticRegression

import categorical2numerical

UNKNOWN_FEATURES = ['job', 'marital', 'education', 'housing', 'loan', 'default', 'poutcome']

def apply_common_imputation(df):
    # Fill unknown attribute values with the common value
    for attribute in UNKNOWN_FEATURES:
        attribute_value_counts = df[attribute].value_counts()
        df.loc[df[attribute] == "unknown", attribute] = attribute_value_counts.idxmax()
        df.loc[df[attribute] == "nonexistent", attribute] = attribute_value_counts.idxmax()

def apply_mean_imputation(df):
    # Fill unknown attribute values with the mean value
    temp_df = df[df.pdays != 999]
    df.loc[df["pdays"] == 999, "pdays"] = temp_df["pdays"].mean()

# TODO IMPLEMENT REGRETION_IMPUTATION
def apply_regression_imputation(df, target_column_name):

    unknown_indexes = []
    train_index_list, test_index_list = [], []
    for index, row in df.iterrows():
        if row[target_column_name] == "unknown" or row[target_column_name] == "nonexistance":
            test_index_list.append(index)
            unknown_indexes.append(index)
        else:
            train_index_list.append(index)

    train_df = copy.copy(df.loc[train_index_list])
    test_df = copy.copy(df.loc[test_index_list])

    target = train_df[target_column_name]

    apply_common_imputation(train_df)
    apply_common_imputation(test_df)

    categorical2numerical.apply_label_encoding(train_df)
    categorical2numerical.apply_label_encoding(test_df)
    train_df = categorical2numerical.apply_onehot_encoding(train_df)
    test_df = categorical2numerical.apply_onehot_encoding(test_df)

    # print(train_df)
    # print(test_df)

    cols_to_drop = []
    for col in train_df.columns:
        if str(col).startswith(target_column_name):
            cols_to_drop.append(col)

    for col in test_df.columns:
        if str(col).startswith(target_column_name):
            cols_to_drop.append(col)

    # print(cols_to_drop)

    for col in cols_to_drop:
        if train_df.__contains__(col):
            del train_df[col]
        if test_df.__contains__(col):
            del test_df[col]

    # print(train_df.columns)
    # print(test_df.columns)

    model = LogisticRegression()
    model.fit(train_df, target)
    missing_values = model.predict(test_df)
    print(len(missing_values))
    print(missing_values)

    # for i in range(len(unknown_indexes)):
    #     df.loc[i] = missing_values[i]

    # for index, row in df.iterrows():
    #     if row[target_column_name] == "unknown" or row[target_column_name] == "nonexistance":
            # print("No")
