import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn

CORRELATION_FILTER = 0.8
best_attributes = []


def apply_univariate_selection(df):
    global attributeScores

    x = df.iloc[:, df.columns != 'y']
    y = df.iloc[:, df.columns == 'y']
    # print(x.shape)

    best_attributes = SelectKBest(score_func=chi2, k=25)
    fit = best_attributes.fit(x, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x.columns)

    attributeScores = pd.concat([dfcolumns, dfscores], axis=1)
    attributeScores.columns = ['Top attributes', 'attribute Score']
    best_attributes = attributeScores.nlargest(25, 'attribute Score')
    print(best_attributes)

    attributes_to_eliminate = []
    for column in df.columns:
        if column not in best_attributes["Top attributes"].tolist() and column != 'y':
            attributes_to_eliminate.append(column)
    print("attributes_to_eliminate: ", attributes_to_eliminate)
    print("len(attributes_to_eliminate): ", len(attributes_to_eliminate))

    for attribute in attributes_to_eliminate:
        del df[attribute]

    # TODO I SHOULD NOT REMOVE Y FROM DATASET
    # return df.iloc[:, cols]
    # return new_df


def show_correlation_matrix(df):
    df_corr = df.corr()
    plt.figure(figsize=(25, 25))
    seaborn.heatmap(df_corr, annot=False)
    plt.show()


def find_correlated_attributes(df):
    df_corr = df.corr()
    correlated_attributes = []
    for i in df_corr:
        for j in df_corr:
            if abs(df_corr[i][j]) >= CORRELATION_FILTER and i != j:
                if not correlated_attributes.__contains__([j, i]):
                    correlated_attributes.append([i, j])
    # correlated attributes
    print("Correlated attributes: " + str(correlated_attributes))

    # number of attributes to be eliminated
    print("Number of attributes to be removed by correlation filter method: " + str(len(correlated_attributes)))
    return correlated_attributes


def apply_correlation_filter_method(df):
    correlated_attributes = find_correlated_attributes(df)
    best_attributes = attributeScores.nlargest(26, 'attribute Score')
    attributes_to_eliminate = []
    for correlated in correlated_attributes:
        for idx, row in best_attributes.iterrows():
            if row["Top attributes"] == correlated[0]:
                score0 = row["attribute Score"]
                column0 = row["Top attributes"]
                idx0 = idx
            elif row["Top attributes"] == correlated[1]:
                score1 = row["attribute Score"]
                column1 = row["Top attributes"]
        # print("Correlated attribute 0: ", idx0, score0)
        # print("Correlated attribute 1: ", idx1, score1)
        if score0 >= score1:
            attributes_to_eliminate.append(str(column1))
        else:
            attributes_to_eliminate.append(str(column0))

    print("---------------------------------- Correlation Filter Method ------------------------------------")
    # print("attributes_to_eliminate", attributes_to_eliminate)
    for attribute in attributes_to_eliminate:
        if attribute in df:
            print("Drop attribute: ", attribute)
            del df[attribute]
    print(df.shape)
    print("-------------------------------------------------------------------------------------------------")
