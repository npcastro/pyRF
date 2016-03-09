# coding=utf-8

# Este script lo ocupe para encontrar el mejor fit para un arbol de decision entre artas combinaciones
# de parametros
# -------------------------------------------------------------------------------------------------

import itertools
import metrics
import pandas as pd
import tree
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

folds = 4

# result_dir = 'Resultados/Fitting/MACHO/'
result_dir = 'Resultados/Fitting/EROS/'

# Parametros a probar
min_samples_splits = range(10, 100, 10)
max_depths = range(8,16, 2)

# params = [a for a in itertools.product(min_samples_splits, max_depths)]
params = xrange(50,1000,20)

for p in params:

    # min_samples_split = p[0]
    # max_depth = p[1]

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
