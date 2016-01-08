# coding=utf-8
# Entrena un random forest en un dataset aumentado y clasifica ocupando las medias
# de los GP como datos de testing

from config import *
import metrics
import utils

import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

import pickle
import sys

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        percentage = sys.argv[1]
    else:
        percentage = '100'

    folds = 10
    training_path = '/n/seasfs03/IACS/TSC/ncastro/sets/MACHO_Big/macho big ' + percentage + '.csv'
    test_path = '/n/home09/ncastro/workspace/Features/sets/MACHO_Means/Macho means set ' + percentage + '.csv'

    # Index Filter??

    feature_filter = ['Amplitude', 'AndersonDarling', 'Autocor_length', 'Beyond1Std', 'Con',
                      'Eta_e', 'LinearTrend', 'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
                      'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'Q31', 'Rcs', 'Skew',
                      'SlottedA_length', 'SmallKurtosis', 'Std', 'StetsonK', 'StetsonK_AC']

    train_data = pd.read_csv(train_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    # Elimino curvas cuyo id este repetido en el set de testing
    test_data = utils.remove_duplicate_index(test_data)

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

    train_X = train_X[feature_filter]
    test_X = test_X[feature_filter]

    skf = cross_validation.StratifiedKFold(y, n_folds=folds)

    results = []
    ids = []

    for train_index, test_index in skf:
        fold_test_X = test_X.iloc[test_index]
        fold_test_y = test_y.iloc[test_index]

        aux_index = test_X.iloc[train_index].index

        fold_train_X = train_X.loc[aux_index]
        fold_train_y = train_y.loc[aux_index]

        clf = None
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=14, min_samples_split=5, n_jobs=-1)

        clf.fit(fold_train_X, fold_train_y)
        results.append(metrics.predict_table(clf, fold_test_X, fold_test_y))
        ids.extend(fold_test_X.index.tolist())

    result = pd.concat(results)
    result['indice'] = ids
    result.set_index('indice')
    result.index.name = None
    result = result.drop('indice', axis=1)

    output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Sampled/Big/Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Sampled/Big/Predicciones/result_' + percentage + '.csv')
