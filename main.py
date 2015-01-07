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
    data['weight'] = data['weight'].astype(float)

    # path = "sets/macho_60.csv"
    data = pd.read_csv(path)

    data = data.dropna(axis=0, how='any')

    skf = cross_validation.StratifiedKFold(data['class'], n_folds=folds)

    results = []
    count = 1
    for train_index, test_index in skf:
        print 'Fold: ' + str(count)
        count += 1
        train, test = data.iloc[train_index], data.iloc[test_index]

        clf = None
        # clf = tree.Tree('confianza')
        # clf = tree.Tree('gain')
        clf = tree.Tree('uncertainty', max_depth=10,
                        min_samples_split=20, most_mass_threshold=0.9, min_mass_threshold=0.10,
                        min_weight_threshold=0.1)

        clf.fit(train)

        results.append(clf.predict_table(test))

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
