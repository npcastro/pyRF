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

        # path = "sets/macho " + str(p) + ".csv"
        path = "sets/macho random.csv"

        # Obtengo los nombres de las variables
        with open(path, 'r') as f:
            nombres = f.readline().strip().split(' ')
        f.close()
        nombres = nombres[0:-1]
        nombres.append('class')


        # data = pd.read_csv(path, sep=' ', header=None, names=nombres, skiprows=1, index_col=0)
        data = pd.read_csv(path)

        data['weight'] = data['weight'].astype(float)

        data = data.dropna(axis = 0, how='any')

        # Para testing rapido
        data = data.iloc[0:1000]

        # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
        
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

        # # Serializo los resultados con pickle
        output = open( 'output/macho/arbol random.pkl', 'w')
        pickle.dump(clf, output)
        output.close()

        output = open( 'output/macho/result random.pkl', 'w')
        pickle.dump(result, output)
        output.close()


        # output = open( 'output/macho/arbol ' + str(p) + '.pkl', 'w')
        # pickle.dump(clf, output)
        # output.close()

        # output = open( 'output/macho/result '+ str(p) + '.pkl', 'w')
        # pickle.dump(result, output)
        # output.close()