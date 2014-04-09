# Toma un set de entrenamiento, entrena un arbol y luego guarda el arbol aprendido y su performance resultados.

import tree
import pandas as pd
import numpy as np
import pickle
from sklearn import cross_validation


if __name__ == '__main__':

    # porcentajes = [20,40,60,80]
    porcentajes = [20]
    folds = 2

    for p in porcentajes:

        # path = "/Users/npcastro/workspace/Features/Entrenamiento var_comp/Entrenamiento " + str(p) + ".txt"
        # path = "/Users/npcastro/workspace/Features/Entrenamiento comp/Entrenamiento " + str(p) + ".txt"
        # path = "/Users/npcastro/workspace/Features/Entrenamiento trust/Entrenamiento " + str(p) + ".txt"
        # path = "/Users/npcastro/workspace/Features/Entrenamiento new_var/Entrenamiento " + str(p) + ".txt"

        path = "sets/u_iris 10.csv"

        # Obtengo los nombres de las variables
        with open(path, 'r') as f:
            nombres = f.readline().strip().split(' ')
        f.close()
        nombres = nombres[0:-1]
        nombres.append('class')


        # data = pd.read_csv(path, sep=' ', header=None, names=nombres, skiprows=1, index_col=0)

        data = pd.read_csv(path)

        data = data.dropna(axis = 0, how='any')

        # Para testing rapido
        # data = data.iloc[0:300]

        # Hago cross validation
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
            clf = tree.Tree('uncertainty')

            clf.fit(train)

            results.append(clf.predict_table(test))

        result = pd.concat(results)
        matrix = clf.confusion_matrix(result)

        # Serializo los resultados con pickle
        
        output = open( 'output/arbol ' + str(p) + '.pkl', 'w')
        pickle.dump(clf, output)
        output.close()

        output = open( 'output/result '+ str(p) + '.pkl', 'w')
        pickle.dump(result, output)
        output.close()