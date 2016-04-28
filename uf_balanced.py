# coding=utf-8

# -------------------------------------------------------------------------------------------------

from functools import partial
from multiprocessing import Pool
import argparse
import sys

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import metrics
import parallel
import utils

if __name__ == '__main__':

    # Recibo parámetros de la linea de comandos
    print ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', required=True, type=str)
    parser.add_argument('--n_samples', required=True, type=int)
    parser.add_argument('--catalog', default='MACHO', choices=['MACHO', 'EROS', 'OGLE'])    

    parser.add_argument('--max_depth',  required=False, type=int)
    parser.add_argument('--min_samples_split',  required=False, type=int)

    parser.add_argument('--sets_path',  required=True, type=str)
    parser.add_argument('--result_path',  required=True, type=str)
    
    parser.add_argument('--train_index_filter', required=False, type=str)
    parser.add_argument('--test_index_filter', required=False, type=str)
    parser.add_argument('--feature_filter',  nargs='*', type=str)

    args = parser.parse_args(sys.argv[1:])

    percentage = args.percentage
    n_samples = args.n_samples
    catalog = args.catalog

    max_depth = args.max_depth
    min_samples_split = args.min_samples_split

    sets_path = args.sets_path
    result_path = args.result_path
    
    train_index_filter = args.train_index_filter
    test_index_filter = args.test_index_filter
    feature_filter = args.feature_filter

    if train_index_filter is not None:
        train_index_filter = pd.read_csv(train_index_filter, index_col=0).index

    if test_index_filter is not None:
        test_index_filter = pd.read_csv(test_index_filter, index_col=0).index

    paths = [sets_path + catalog + '_sampled_' + str(i) + '.csv' for i in xrange(n_samples)]

    resultados = []
    for p in paths:
        data = pd.read_csv(p, index_col=0)

        train_X, train_y = utils.filter_data(data, index_filter=train_index_filter, feature_filter=feature_filter)
        test_X, test_y = utils.filter_data(data, index_filter=test_index_filter, feature_filter=feature_filter)

        clf = None
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth,
                                     min_samples_split=min_samples_split)

        clf.fit(train_X, train_y)
        resultados.append(metrics.predict_table(clf, test_X, test_y))

    result = metrics.aggregate_predictions(resultados)
    result.to_csv(result_path + 'result_' + percentage + '.csv')

    print metrics.weighted_f_score(metrics.confusion_matrix(result))
