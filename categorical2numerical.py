
import pandas as pd

# pd.set_option('display.max_columns', 100)

from sklearn import preprocessing

def apply_label_encoding(df):
    # binary attributes
    label_encoder = preprocessing.LabelEncoder()
    df.housing = label_encoder.fit_transform(df.housing)
    df.loan = label_encoder.fit_transform(df.loan)
    df.default = label_encoder.fit_transform(df.default)
    df.contact = label_encoder.fit_transform(df.contact)
    # df.y = label_encoder.fit_transform(df.y)

def apply_onehot_encoding(df):
    new_df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'poutcome', 'month', 'day_of_week'])
    return new_df
    # print(df.head())


