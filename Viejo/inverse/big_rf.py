# coding=utf-8
# Entrena un random forest en un dataset aumentado y clasifica ocupando las medias
# de los GP como datos de testing

# -------------------------------------------------------------------------------------------------

import argparse
import sys

import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

import metrics
import utils



if __name__ == '__main__':

    print ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_processes', required=True, type=int)
    parser.add_argument('--catalog', default='MACHO', choices=['MACHO', 'EROS', 'OGLE'])

    parser.add_argument('--train_path',  required=True, type=str)
    parser.add_argument('--test_path',  required=True, type=str)
    parser.add_argument('--result_path',  required=True, type=str)
    parser.add_argument('--n_estimators', required=False, type=int)
    parser.add_argument('--criterion', required=False, type=str)
    parser.add_argument('--max_depth', required=False, type=int)
    parser.add_argument('--min_samples_split', required=False, type=int)
    parser.add_argument('--feature_filter',  nargs='*', type=str)

    args = parser.parse_args(sys.argv[1:])

    n_processes = args.n_processes
    catalog = args.catalog

    train_path = args.train_path
    test_path = args.test_path
    result_path = args.result_path
    n_estimators = args.n_estimators
    criterion = args.criterion
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split
    feature_filter = args.feature_filter

    train_data = pd.read_csv(train_path, index_col=0)
    train_index_filter = pd.read_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/RF/Small/train.csv', index_col=0).index
    train_X, train_y = utils.filter_data(train_data, index_filter=train_index_filter, feature_filter=feature_filter)

    test_data = pd.read_csv(test_path, index_col=0)
    test_index_filter = pd.read_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/RF/Small/test.csv', index_col=0).index
    test_X, test_y = utils.filter_data(test_data, index_filter=test_index_filter, feature_filter=feature_filter)

    results = []
    ids = []

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
    result.index.name = None
    result = result.drop('indice', axis=1)

    result.to_csv(result_path)

    m = metrics.confusion_matrix(result)
    print metrics.weighted_f_score(m)
