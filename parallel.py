# coding=utf-8
import tree

import pandas as pd
from sklearn import cross_validation

def fit_tree(path, index_filter=None, class_filter=None, feature_filter=None, folds=10):
    """

    path: Dirección del dataset a ocupar para entrenar
    index_filter: Pandas index para filtrar las filas del dataset que se quieren utilizar
    class_filter: Lista de clases que se quiere utilizar
    feature_filter: Lista de features que se quiere utilizar

    """
    data = pd.read_csv(path, index_col=0)
    
    if index_filter:
        data = data.loc[index_filter]
    
    if class_filter:
        data = data[data['class'].apply(lambda x: True if x in class_filter else False)]

    data = data.dropna(axis=0, how='any')
    y = data['class']
    data = data.drop('class', axis=1)

    if feature_filter:
        data = data[feature_filter]

    skf = cross_validation.StratifiedKFold(y, n_folds=folds)
    
    results = []
    for train_index, test_index in skf:
        train_X, test_X = data.iloc[train_index], data.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        clf = None
        clf = tree.Tree('gain', max_depth=10, min_samples_split=20)

        clf.fit(train_X, train_y)
        result = clf.predict_table(test_X, test_y)
        results.append(result)
        

    return pd.concat(results)