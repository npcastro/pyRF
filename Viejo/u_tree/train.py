# coding=utf-8

# Entrena un arbol de decisión con incertidumbre en paralelo
# -------------------------------------------------------------------------------------------------

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

    folds = 5
    training_set_path = '/n/home09/ncastro/workspace/Features/sets/MACHO_GP/gp_u_set_' + percentage + '.csv'
    data = pd.read_csv(training_set_path, index_col=0)
    data = data[data['class'].apply(lambda x: True if x in ['Be_lc','EB'] else False)]

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

    output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Comparacion/UTree/Arboles/Arbol GP_' + percentage + '.pkl', 'wb+')
    #output = open('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/UTree/GP/Arboles/Arbol GP_' + percentage + '.pkl', 'wb+')
    pickle.dump(clf, output)
    output.close()

    result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Comparacion/UTree/Predicciones/result_' + percentage + '.csv')
    #result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/UTree/GP/Predicciones/result_' + percentage + '.csv')
