# coding=utf-8
# Entra un arbol de decisión con incertidumbre en paralelo
# Y guarda sus resultados

from config import *
import tree

import pandas as pd
from sklearn import cross_validation

import pickle
import sys

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        percentage = sys.argv[1]
    else:
        percentage = '100'

    n_jobs = 30

    folds = 10
    training_set_path = SETS_DIR_PATH + 'GP/gp_u_set_' + percentage + '.csv'
    data = pd.read_csv(training_set_path)
    # data = data.iloc[0:1500]

    data = data.dropna(axis=0, how='any')
    data['weight'] = data['weight'].astype(float)
    y = data['class']

    X = data.drop('class', axis=1)

    skf = cross_validation.StratifiedKFold(y, n_folds=folds)

    results = []
    count = 1

    for train_index, test_index in skf:
        print 'Fold: ' + str(count)
        count += 1

        train_X, test_X = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        clf = None

        clf = tree.Tree('uncertainty', max_depth=10, min_samples_split=20,
                        most_mass_threshold=0.9, min_mass_threshold=0.1,
                        min_weight_threshold=0.01, parallel='features',
                        n_jobs=n_jobs)
        
        clf.fit(train_X, train_y)
        results.append(clf.predict_table(test_X, test_y))

    result = pd.concat(results)

    output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/GP/Arboles/Arbol GP_' + percentage + '.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/GP/Predicciones/result_' + percentage + '.csv')
