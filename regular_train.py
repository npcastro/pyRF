# coding=utf-8

# Entrena un arbol de decisión clasico y guarda sus resultados
# ------------------------------------------------------------------------------------------------- 

import argparse
import pickle
import sys

import pandas as pd
from sklearn import cross_validation

from config import *
import tree
import utils

if __name__ == '__main__':

    # Recibo parámetros de la linea de comandos
    print ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', required=True, type=str)
    parser.add_argument('--folds',  required=True, type=int)
    parser.add_argument('--training_set_path',  required=True, type=str)
    parser.add_argument('--result_path',  required=True, type=str)
    parser.add_argument('--class_filter',  nargs='*', type=str)
    parser.add_argument('--feature_filter',  nargs='*', type=str)

    args = parser.parse_args(sys.argv[1:])

    percentage = args.percentage
    folds = args.folds
    training_set_path = args.training_set_path
    result_path = args.result_path
    class_filter = args.class_filter
    feature_filter = args.feature_filter


    data = pd.read_csv(training_set_path, index_col=0)
    data, y = utils.filter_data(data, index_filter=None, class_filter=class_filter,
                                feature_filter=feature_filter)

    skf = cross_validation.StratifiedKFold(y, n_folds=folds)

    results = []
    ids = []
    for train_index, test_index in skf:
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

    output = open(result_path + 'Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    result.to_csv(result_path + '/Predicciones/result_' + percentage + '.csv')

