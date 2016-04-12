# coding=utf-8
# Entra un arbol de decisión clasico sobre los sets de entrenamiento con curvas
# completadas

import tree

import pandas as pd
from sklearn import cross_validation

import pickle
import sys

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        total_points = sys.argv[1]
    else:
        total_points = '100'

    percentage = 20

    folds = 10
    training_set_path = '/n/seasfs03/IACS/TSC/ncastro/GP_Sets/EROS/' + str(percentage) + '%/EROS_completed_set_' + str(total_points) + '.csv'

    data = pd.read_csv(training_set_path, index_col=0)

    data = data.dropna(axis=0, how='any')
    y = data['class']

    data = data.drop('class', axis=1)

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

    output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/EROS/Tree/Completed/Arboles/Arbol_' + str(total_points) + '.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/EROS/Tree/Completed/Predicciones/result_' + str(total_points) + '.csv')

