from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def apply_min_max(df):
    x = df.iloc[:, df.columns != 'y']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x)
    x = scaler.transform(x)
    df.iloc[:, df.columns != 'y'] = x


def apply_standard_scalar(df):
    x = df.iloc[:, df.columns != 'y']
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    df.iloc[:, df.columns != 'y'] = x
