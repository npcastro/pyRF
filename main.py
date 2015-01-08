# Toma un set de entrenamiento, entrena un arbol y luego guarda el arbol aprendido y su performance
# resultados.

import tree
import pandas as pd
import numpy as np
import pickle
from sklearn import cross_validation


if __name__ == '__main__':

    folds = 10
    path = "sets/gp_u_set_60.csv"
    data = pd.read_csv(path)
    data = data.dropna(axis=0, how='any')
    data['weight'] = data['weight'].astype(float)
    # skf = cross_validation.StratifiedKFold(data['class'], n_folds=folds)

    # path = "sets/macho_60.csv"
    # data = pd.read_csv(path)
    # data = data.dropna(axis=0, how='any')
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
        # train, test = data.iloc[train_index], data.iloc[test_index]

        clf = None
        # clf = tree.Tree('gain')
        # clf.fit(train_X, train_y)

        clf = tree.Tree('uncertainty', max_depth=10,
                        min_samples_split=20, most_mass_threshold=0.9, min_mass_threshold=0.10,
                        min_weight_threshold=0.01)

        clf.fit(train_X, train_y)

        # clf.fit(train)

        results.append(clf.predict_table(test_X, test_y))
        # results.append(clf.predict_table(test.drop('class', axis=1), test['class']))

        break

    result = pd.concat(results)
    matrix = clf.confusion_matrix(result)

    # Serializo los resultados con pickle
    output = open('Resultados/GP/Fixed George/Arbol GP.pkl', 'w')
    pickle.dump(clf, output)
    output.close()

    output = open('Resultados/GP/Fixed George/result.pkl', 'w')
    pickle.dump(result, output)
    output.close()

    clases = matrix.columns.tolist()
    p = [clf.precision(matrix, c) for c in clases]
    r = [clf.recall(matrix, c) for c in clases]
    f = [clf.f_score(matrix, c) for c in clases]
