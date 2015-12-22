# coding=utf-8
# Entra un arbol de decisión clasico
# Y guarda sus resultados

from config import *
import tree

import pandas as pd
from sklearn import cross_validation

import pickle
import sys

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        percentage = sys.argv[1]
    else:
        percentage = '100'

    folds = 5

    training_set_path = SETS_DIR_PATH + 'MACHO_Reduced/Macho reduced set ' + percentage + '.csv'
    #training_set_path = SETS_DIR_PATH + 'MACHO/Macho regular set ' + percentage + '.csv'
    # training_set_path = '/n/home09/ncastro/workspace/pyRF/sets/MACHO random II/Macho random ' + percentage + '.csv'
    # training_set_path = '/n/home09/ncastro/workspace/pyRF/sets/EROS random II/EROS random ' + percentage + '.csv'
    # training_set_path = '/n/home09/ncastro/workspace/Features/sets/EROS/EROS regular set ' + percentage + '.csv'

    data = pd.read_csv(training_set_path, index_col=0)
    # data = pd.read_csv(training_set_path)

    # Filtro para dejar solo las clases malas
    data = data[data['class'].apply(lambda x: True if x in ['Be_lc','EB'] else False)]

    data = data.dropna(axis=0, how='any')
    y = data['class']

    if 'weight' in data.columns.tolist():
        data = data.drop('weight', axis=1)

    data = data.drop('class', axis=1)
    data = data[['Amplitude', 'AndersonDarling', 'Autocor_length', 'Beyond1Std', 'Con', 'Eta_e', 'LinearTrend',
                 'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev', 'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude',
                 'Q31', 'Rcs', 'Skew', 'SlottedA_length', 'SmallKurtosis', 'Std', 'StetsonK', 'StetsonK_AC']]

    skf = cross_validation.StratifiedKFold(y, n_folds=folds)

    results = []
    ids = []
    count = 1

    for train_index, test_index in skf:
        print 'Fold: ' + str(count)
        count += 1

        train_X, test_X = data.iloc[train_index], data.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        clf = None

        clf = tree.Tree('gain', max_depth=10, min_samples_split=20)

        clf.fit(train_X, train_y)
        results.append(clf.predict_table(test_X, test_y))
        ids.extend(test_X.index.tolist())

    result = pd.concat(results)
    result['indice'] = ids
    result.set_index('indice')
    result.index.name = None
    result = result.drop('indice', axis=1)

    # output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/EROS/Tree/Regular/Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    # output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Tree/Regular/Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Comparacion/Tree/Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    # result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/EROS/Tree/Regular/Predicciones/result_' + percentage + '.csv')
    # result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Tree/Regular/Predicciones/result_' + percentage + '.csv')
    result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Comparacion/Tree/Predicciones/result_' + percentage + '.csv')

