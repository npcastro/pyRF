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

    folds = 10
    training_set_path = SETS_DIR_PATH + 'Macho regular set ' + percentage + '.csv'
    data = pd.read_csv(training_set_path)

    data = data.dropna(axis=0, how='any')
    y = data['class']

    skf = cross_validation.StratifiedKFold(y, n_folds=folds)

    results = []
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

    result = pd.concat(results)

    output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/Regular/Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/Regular/Predicciones/result_' + percentage + '.csv')

