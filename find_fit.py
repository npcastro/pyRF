# coding=utf-8

# Este script lo ocupe para encontrar el mejor fit para un arbol de decision entre artas combinaciones
# de parametros
# -------------------------------------------------------------------------------------------------

import itertools
import argparse
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
import pandas as pd

import metrics


if __name__ == '__main__':

    # Recibo parámetros de la linea de comandos
    print ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_processes', required=True, type=int)
    parser.add_argument('--folds', required=True, type=int)
    parser.add_argument('--model', default='sktree', choices=['rf', 'sktree'])
    parser.add_argument('--catalog', default='MACHO', choices=['MACHO', 'EROS', 'OGLE'])
    parser.add_argument('--n_samples', required=False, default=100, type=int)

    parser.add_argument('--sets_path',  required=True, type=str)
    parser.add_argument('--result_dir',  required=True, type=str)

    parser.add_argument('--feature_filter',  nargs='*', type=str)

    args = parser.parse_args(sys.argv[1:])

    n_processes = args.n_processes
    folds = args.folds
    model = args.model
    catalog = args.catalog
    n_samples = args.n_samples

    sets_path = args.sets_path
    result_dir = args.result_dir

    feature_filter = args.feature_filter

    # Parametros a probar
    min_samples_splits = range(2, 20, 2)
    max_depths = range(8,16, 2)

    params = [a for a in itertools.product(min_samples_splits, max_depths)]

    for min_samples_split, max_depth in params:

        # path = '/Users/npcastro/workspace/Features/sets/MACHO/Macho regular set 40.csv'
        path = '/Users/npcastro/workspace/Features/sets/EROS/EROS regular set 40.csv'
        data = pd.read_csv(path)

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
            clf = RandomForestClassifier(n_estimators=p, criterion='entropy',
                                         max_depth=14, min_samples_split=20,
                                         n_jobs=2)
            
            clf.fit(train_X, train_y)
            results.append(metrics.predict_table(clf, test_X, test_y))
            

        result = pd.concat(results)

        matrix = metrics.confusion_matrix(result)

        clases = matrix.columns.tolist()
        precisions = [metrics.precision(matrix, c) for c in clases]
        recalls = [metrics.recall(matrix, c) for c in clases]
        f_scores = [metrics.f_score(matrix, c) for c in clases]

        w_score = metrics.weighted_f_score(matrix)

        # f = open(result_dir + str(max_depth) + ' ' + str(min_samples_split) + '.txt', 'w')
        f = open(result_dir + str(p) + '.txt', 'w')

        f.write('F_score by class')
        f.write('\n')
        f.write(str(f_scores))
        f.write('\n')
        f.write('\n')
        f.write('Weighted average: ')
        f.write(str(w_score))

        f.close()
