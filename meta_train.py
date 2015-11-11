# coding=utf-8
# Hace una meta clasificación con arboles de clasificacion clasicos y con incertidumbre

from config import *
import tree

import pandas as pd
from sklearn import cross_validation

import pickle
import sys


def meta_datasets(data, list_B, list_C):

    # Los datos dificiles de clasificar (B)
    data_b = data[data['class'].apply(lambda x: True if x in list_B else False)]

    # Los datos faciles de clasificar (C)
    data_c = data[data['class'].apply(lambda x: True if x not in list_C else False)]

    # El set con solo dos clases B y C
    y = data['class']
    y_a = y.apply(lambda x: 'B' if x in list_B else 'C')
    data['class'] = y_a

    return data, data_b, data_c

if __name__ == '__main__':

    folds = 10
    training_set_path = '/Users/npcastro/workspace/Features/sets/MACHO/Macho regular set 40.csv'
    list_B = ['non_variables', 'CEPH', 'quasar_lc', 'longperiod_lc', 'RRL', 'microlensing_lc']
    list_C = ['Be_lc','EB']

    data = pd.read_csv(training_set_path)
    data = data[['Amplitude', 'AndersonDarling', 'Autocor_length', 'Beyond1Std', 'Con', 'Eta_e', 'LinearTrend',
                 'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev', 'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude',
                 'Q31', 'Rcs', 'Skew', 'SlottedA_length', 'SmallKurtosis', 'Std', 'StetsonK', 'StetsonK_AC', 'class']]
    
    data = data.dropna(axis=0, how='any')
    y = data['class']
    
    skf = cross_validation.StratifiedKFold(y, n_folds=folds)

    results = []
    count = 1

    for train_index, test_index in skf:
        print 'Fold: ' + str(count)
        count += 1

        # train_X, test_X = data.iloc[train_index], data.iloc[test_index]
        # train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        train, test = data.iloc[train_index], data.iloc[test_index]

        train_a, train_b, train_c = meta_datasets(train, list_B, list_C)
        y_a = train_a['class']
        train_a.drop('class', axis=1, inplace=True)
        y_b = train_b['class']
        train_b.drop('class', axis=1, inplace=True)
        y_c = train_c['class']
        train_c.drop('class', axis=1, inplace=True)

        clf_a = None
        clf_b = None
        clf_c = None

        clf_a = tree.Tree('gain', max_depth=10, min_samples_split=20)
        clf_b = tree.Tree('uncertainty', max_depth=10, min_samples_split=20,
                          most_mass_threshold=0.9, min_mass_threshold=0.1,
                          min_weight_threshold=0.01, parallel='features',
                          n_jobs=n_jobs)
        clf_c = tree.Tree('gain', max_depth=10, min_samples_split=20)

        clf_a.fit(train_a, y_a)
        clf_b.fit(train_b, y_b)
        clf_c.fit(train_c, y_c)

        results.append(clf.predict_table(test_X, test_y))

    result = pd.concat(results)

    # output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/EROS/Tree/Regular/Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Tree/Regular/Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    # result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/EROS/Tree/Regular/Predicciones/result_' + percentage + '.csv')
    result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Tree/Regular/Predicciones/result_' + percentage + '.csv')

