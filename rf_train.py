# coding=utf-8
# Entrena un random forest
# Y guarda sus resultados

from config import *
import metrics

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
    training_set_path = SETS_DIR_PATH + 'MACHO/Macho regular set ' + percentage + '.csv'
    # training_set_path = '/n/home09/ncastro/workspace/Features/sets/EROS/EROS regular set ' + percentage + '.csv'

    data = pd.read_csv(training_set_path)
    data = data.dropna(axis=0, how='any')
    y = data['class']
    data = data.drop('class', axis=1)
    #data = data[['Amplitude', 'AndersonDarling', 'Autocor_length', 'Beyond1Std', 'Con', 'Eta_e', 'LinearTrend',
    #             'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev', 'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude',
    #             'Q31', 'Rcs', 'Skew', 'SlottedA_length', 'SmallKurtosis', 'Std', 'StetsonK', 'StetsonK_AC']]

    skf = cross_validation.StratifiedKFold(y, n_folds=folds)

    results = []
    count = 1

    for train_index, test_index in skf:
        print 'Fold: ' + str(count)
        count += 1

        train_X, test_X = data.iloc[train_index], data.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        clf = None
        clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=14, min_samples_split=20, n_jobs=-1)

        clf.fit(train_X, train_y)
        results.append(metrics.predict_table(clf, test_X, test_y))

    result = pd.concat(results)

    # output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/EROS/RF/Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/RF/Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    # result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/EROS/RF/Predicciones/result_' + percentage + '.csv')
    result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/RF/Predicciones/result_' + percentage + '.csv')

