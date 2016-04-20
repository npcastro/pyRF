# coding=utf-8

# Entrena un random forest y guarda sus resultados
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
    parser.add_argument('--n_processes', required=True, type=int)
    parser.add_argument('--catalog', default='MACHO', choices=['MACHO', 'EROS', 'OGLE'])
    parser.add_argument('--folds', required=True, type=int)
    parser.add_argument('--training_set_path', required=True, type=str)
    parser.add_argument('--result_path', required=True, type=str)
    parser.add_argument('--lc_filter', required=False, type=float, 
                        help='Percentage of the total amount of data to use')
    parser.add_argument('--n_estimators', required=False, type=int)
    parser.add_argument('--criterion', required=False, type=str)
    parser.add_argument('--max_depth', required=False, type=int)
    parser.add_argument('--min_samples_split', required=False, type=int)
    parser.add_argument('--index_filter', required=False, type=str)
    parser.add_argument('--feature_filter',  nargs='*', type=str)

    args = parser.parse_args(sys.argv[1:])
    
    percentage = args.percentage
    catalog = args.catalog
    n_processes = args.n_processes
    folds = args.folds
    training_set_path = args.training_set_path
    result_path = args.result_path
    lc_filter = args.lc_filter
    n_estimators = args.n_estimators
    criterion = args.criterion
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split
    index_filter = args.index_filter
    feature_filter = args.feature_filter

    data = pd.read_csv(training_set_path, index_col=0)

    if index_filter is not None:
        index_filter = pd.read_csv(index_filter, index_col=0).index

    elif lc_filter is not None:
        # Filtro un porcentaje de las curvas y las guardo para comparar despues
        data = utils.stratified_filter(data, data['class'], lc_filter)
        data.to_csv(result_path + 'data.csv')

    data, y = utils.filter_data(data, feature_filter=feature_filter, index_filter=index_filter)

    skf = cross_validation.StratifiedKFold(y, n_folds=folds)

    results = []
    ids = []

    for train_index, test_index in skf:

        train_X, test_X = data.iloc[train_index], data.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        clf = None
        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                     max_depth=max_depth, min_samples_split=min_samples_split,
                                     n_jobs=n_processes)

        clf.fit(train_X, train_y)
        results.append(metrics.predict_table(clf, test_X, test_y))
        ids.extend(test_X.index.tolist())

    result = pd.concat(results)
    result['indice'] = ids
    result.set_index('indice')
    result.index.name = catalog + '_id'
    result = result.drop('indice', axis=1)

    output = open(result_path + 'Arboles/Arbol_' + percentage + '.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    result.to_csv(result_path + 'Predicciones/result_' + percentage + '.csv')

    print metrics.weighted_f_score(metrics.confusion_matrix(result))
