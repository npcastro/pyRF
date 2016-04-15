# coding=utf-8

# Como mean_forest solo que al reves. Entreno un solo modelo con las medias del GP, pero
# luego al clasificar, se pasan todas las muestras de la curva y se agregan las votaciones

# -------------------------------------------------------------------------------------------------

import argparse
import pickle
import sys

import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

import metrics
import utils

if __name__ == '__main__':

    print ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', required=True, type=str)
    parser.add_argument('--catalog', default='MACHO', choices=['MACHO', 'EROS', 'OGLE'])
    parser.add_argument('--n_processes', required=True, type=int)
    parser.add_argument('--training_set_path', required=True, type=str)
    parser.add_argument('--test_path',  required=True, type=str)
    parser.add_argument('--folds', required=True, type=int)
    parser.add_argument('--n_estimators', required=False, type=int)
    parser.add_argument('--criterion', required=False, type=str)
    parser.add_argument('--max_depth', required=False, type=int)
    parser.add_argument('--min_samples_split', required=False, type=int)
    parser.add_argument('--result_path', required=True, type=str)
    parser.add_argument('--feature_filter',  nargs='*', type=str)

    args = parser.parse_args(sys.argv[1:])
    
    percentage = args.percentage
    catalog = args.catalog
    n_processes = args.n_processes
    training_set_path = args.training_set_path
    folds = args.folds
    test_path = args.test_path
    n_estimators = args.n_estimators
    criterion = args.criterion
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split
    result_path = args.result_path
    feature_filter = args.feature_filter

    data = pd.read_csv(training_set_path, index_col=0)
    data, y = utils.filter_data(data, feature_filter=feature_filter)

    paths = [test_path + catalog + '_sampled_' + str(i) + '.csv' for i in xrange(100)]

    skf = cross_validation.StratifiedKFold(y, n_folds=folds)
    results = []
    ids = []

    for train_index, test_index in skf:

        train_X, train_y  = data.iloc[train_index], y.iloc[train_index]

        clf = None
        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                     max_depth=max_depth, min_samples_split=min_samples_split,
                                     n_jobs=n_processes)
        clf.fit(train_X, train_y)

        aux = []
        for path in paths:
        	test_data = pd.read_csv(path, index_col=0)
        	test_data, test_y = utils.filter_data(test_data, feature_filter=feature_filter)
        	test_X, test_y  = test_data.iloc[train_index], test_y.iloc[train_index]

        	aux.append(metrics.predict_table(clf, test_X, test_y))

        results.append(metrics.aggregate_predictions(aux))
        ids.extend(test_X.index.tolist())

    result = pd.concat(results)

    output = open(result_path + 'Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    result.to_csv(result_path + 'Predicciones/result_' + percentage + '.csv')