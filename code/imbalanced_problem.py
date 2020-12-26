
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import numpy as np
from sklearn import model_selection


def apply_smote(df):
    x = df.iloc[:, df.columns != 'y']
    y = df.iloc[:, df.columns == 'y']

    smt = SMOTE(kind='regular', random_state=42)
    x, y = smt.fit_sample(x, y.values.ravel())
    print(np.bincount(y))
    return x, y

def apply_near_miss(df):
    x = df.iloc[:, df.columns != 'y']
    y = df.iloc[:, df.columns == 'y']

    near_miss = NearMiss()
    x, y = near_miss.fit_sample(x, y.values.ravel())
    # print(np.bincount(y))
    return x, y
