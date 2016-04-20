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
    parser.add_argument('--folds', required=True, type=int)
    parser.add_argument('--inverse', required=False, action='store_true')

    parser.add_argument('--train_path',  required=True, type=str)
    parser.add_argument('--test_path',  required=True, type=str)
    parser.add_argument('--result_path',  required=True, type=str)

    parser.add_argument('--n_estimators', required=False, type=int)
    parser.add_argument('--criterion', required=False, type=str)
    parser.add_argument('--max_depth', required=False, type=int)
    parser.add_argument('--min_samples_split', required=False, type=int)

    parser.add_argument('--feature_filter',  nargs='*', type=str)
    parser.add_argument('--index_filter', required=False, type=str)

    args = parser.parse_args(sys.argv[1:])

    n_processes = args.n_processes
    catalog = args.catalog
    folds = args.folds
    inverse = args.inverse

    train_path = args.train_path
    test_path = args.test_path
    result_path = args.result_path

    n_estimators = args.n_estimators
    criterion = args.criterion
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split

    feature_filter = args.feature_filter
    index_filter = args.index_filter


    index_filter = pd.read_csv(index_filter, index_col=0).index

    train_data = pd.read_csv(train_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    train_data, test_data = utils.equalize_indexes(train_data, test_data)

    train_X, train_y = utils.filter_data(train_data, index_filter=index_filter, feature_filter=feature_filter)
    test_X, test_y = utils.filter_data(test_data, index_filter=index_filter, feature_filter=feature_filter)

    # Ocupo solo los datos de test para hacer el k-fold, por que estos no estan repetidos
    # Y es valido ocuparlos solo por posicion
    skf = cross_validation.StratifiedKFold(test_y, n_folds=folds)
    results = []
    ids = []

    for train_index, test_index in skf:
        if inverse:
            aux = train_index
            train_index = test_index
            test_index = aux

        fold_test_X = test_X.iloc[test_index]
        fold_test_y = test_y.iloc[test_index]

        fold_train_X = train_X.loc[test_X.iloc[train_index].index]
        fold_train_y = train_y.loc[test_y.iloc[train_index].index]

        clf = None
        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                     max_depth=max_depth, min_samples_split=min_samples_split,
                                     n_jobs=n_processes)

        clf.fit(fold_train_X, fold_train_y)
        results.append(metrics.predict_table(clf, fold_test_X, fold_test_y))
        ids.extend(fold_test_X.index.tolist())

    result = pd.concat(results)
    result['indice'] = ids
    result.set_index('indice')
    result.index.name = None
    result = result.drop('indice', axis=1)

    result.to_csv(result_path)

    m = metrics.confusion_matrix(result)
    print metrics.weighted_f_score(m)
