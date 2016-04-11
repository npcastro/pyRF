# coding=utf-8

# Métodos para paralelizar 
# -------------------------------------------------------------------------------------------------

import tree
import utils

import pandas as pd
from sklearn import cross_validation

def train_tree(path, feature_filter=None, train_index=None):
    data = pd.read_csv(path, index_col=0)
    data, y = utils.filter_data(data, feature_filter=feature_filter)
    
    train_X = data.iloc[train_index]
    train_y = y.iloc[train_index]

    clf = None
    clf = tree.Tree('gain', max_depth=10, min_samples_split=20)

    clf.fit(train_X, train_y)
    
    return clf

def fit_montecarlo_tree(path_index, paths = None, index_filter=None, class_filter=None,
                        feature_filter=None, folds=10):
    """A diferencia de fit tree, este metodo recibe todos los paths. Entrena solo con uno, indicado
    por path index. Pero luego por orden, voy abriendo todos los sets para clasificar.
    """
    data = pd.read_csv(paths[path_index], index_col=0)
    data, y = utils.filter_data(data, index_filter, class_filter, feature_filter)

    skf = cross_validation.StratifiedKFold(y, n_folds=folds)

    results = []
    for train_index, test_index in skf:
        train_X = data.iloc[train_index]
        train_y = y.iloc[train_index]

        clf = None
        clf = tree.Tree('gain', max_depth=10, min_samples_split=20)

        clf.fit(train_X, train_y)
        # result = clf.predict_table(test_X, test_y)
        # results.append(result)

    # Ahora clasifico con este arbol para todos los datasets
    for path in paths:
        data = pd.read_csv(path, index_col=0)
        data, y = utils.filter_data(data, index_filter, class_filter, feature_filter)

    return pd.concat(results)
    

def fit_tree(path, index_filter=None, class_filter=None, feature_filter=None, folds=10):
    """

    path: Dirección del dataset a ocupar para entrenar
    index_filter: Pandas index para filtrar las filas del dataset que se quieren utilizar
    class_filter: Lista de clases que se quiere utilizar
    feature_filter: Lista de features que se quiere utilizar

    """
    data = pd.read_csv(path, index_col=0)
    data, y = utils.filter_data(data, index_filter, class_filter, feature_filter)

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

def fit_means_tree(train_path, test_path, index_filter=None, class_filter=None, feature_filter=None, folds=10):
    """

    train_path: Dirección del dataset a ocupar para entrenar
    test_path: Dirección del dataset a ocupar para testear
    index_filter: Pandas index para filtrar las filas del dataset que se quieren utilizar
    class_filter: Lista de clases que se quiere utilizar
    feature_filter: Lista de features que se quiere utilizar

    """
    train_data = pd.read_csv(train_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    # Elimino curvas que estan repetidas
    test_data = utils.remove_duplicate_index(test_data)
    train_data = utils.remove_duplicate_index(train_data)

    if index_filter:
        train_data = train_data.loc[index_filter]
        test_data = test_data.loc[index_filter]
    
    if class_filter:
        train_data = train_data[train_data['class'].apply(lambda x: True if x in class_filter else False)]
        test_data = test_data[test_data['class'].apply(lambda x: True if x in class_filter else False)]

    train_data = train_data.dropna(axis=0, how='any')
    test_data = test_data.dropna(axis=0, how='any')

    # Me aseguro que los datasets sean de los mismos datos
    common_index = list(set(test_data.index.tolist()) & set(train_data.index.tolist()))
    test_data = test_data.loc[common_index]
    train_data = train_data.loc[common_index]
    train_data = train_data.sort_index()
    test_data = test_data.sort_index()

    # Separo features de las clases
    train_y = train_data['class']
    train_X = train_data.drop('class', axis=1)

    test_y = test_data['class']
    test_X = test_data.drop('class', axis=1)

    if feature_filter:
        train_X = train_X[feature_filter]
        test_X = test_X[feature_filter]

    skf = cross_validation.StratifiedKFold(train_y, n_folds=folds)
    
    results = []
    for train_index, test_index in skf:
        fold_train_X = train_X.iloc[train_index]
        fold_train_y = train_y.iloc[train_index]

        fold_test_X = test_X.iloc[test_index]
        fold_test_y = test_y.iloc[test_index]

        clf = None
        clf = tree.Tree('gain', max_depth=10, min_samples_split=20)

        clf.fit(fold_train_X, fold_train_y)

        result = clf.predict_table(fold_test_X, fold_test_y)
        results.append(result)

    return pd.concat(results)
