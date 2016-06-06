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

if __name__ == '__main__':

    print ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_processes', required=False, default=-1, type=int)
    parser.add_argument('--folds', required=True, type=int)

    parser.add_argument('--training_set_path', required=True, type=str)
    parser.add_argument('--result_path', required=True, type=str)
    
    parser.add_argument('--n_estimators', required=False, type=int)
    parser.add_argument('--criterion', required=False, type=str)
    parser.add_argument('--max_depth', required=False, type=int)
    parser.add_argument('--min_samples_split', required=False, type=int)

    args = parser.parse_args(sys.argv[1:])
    
    n_processes = args.n_processes
    folds = args.folds

    training_set_path = args.training_set_path
    result_path = args.result_path

    n_estimators = args.n_estimators
    criterion = args.criterion
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split

    data = pd.read_csv(training_set_path, index_col=0)
    y = data['class']
    data = data.drop('class', axis=1)

    results = []
    skf = cross_validation.StratifiedKFold(y, n_folds=folds)
    for train_index, test_index in skf:
        train_X, test_X = data.iloc[train_index], data.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        clf = None
        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                     max_depth=max_depth, min_samples_split=min_samples_split,
                                     n_jobs=n_processes)

        clf.fit(train_X, train_y)
        results.append(metrics.predict_table(clf, test_X, test_y))

    result = pd.concat(results)

    output = open(result_path + 'Arboles/Arbol.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    result.to_csv(result_path + 'Predicciones/result.csv')

    matrix = metrics.confusion_matrix(result)
    matrix.to_csv(result_path + 'Metricas/soft_matrix_.csv')

    clases = matrix.columns.tolist()
    f_score = [metrics.f_score(matrix, c) for c in clases]

    with open(result_path + 'Metricas/results.txt') as f:
        f.write(clases + '\n')
        f.write(str(f_score) + '\n')