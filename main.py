import tree
import pandas as pd
import numpy as np
import pickle
from sklearn import cross_validation


if __name__ == '__main__':

<<<<<<< HEAD
    # path = "/Users/npcastro/workspace/Features/Resultados 20.txt"
    # path = "/Users/npcastro/workspace/Features/Resultados 40.txt"
    #path = "/Users/npcastro/workspace/Features/Resultados 60.txt"
    # path = "/Users/npcastro/workspace/Features/Resultados 80.txt"
    # path = "/Users/npcastro/workspace/Features/Resultados 100.txt"
=======
    porcentajes = [20,40,60,80]

    for p in porcentajes:

        path = "/Users/npcastro/workspace/Features/Entrenamiento var_comp/Entrenamiento " + str(p) + ".txt"
        # path = "/Users/npcastro/workspace/Features/Entrenamiento comp/Entrenamiento " + str(p) + ".txt"
        # path = "/Users/npcastro/workspace/Features/Entrenamiento trust/Entrenamiento " + str(p) + ".txt"
        # path = "/Users/npcastro/workspace/Features/Entrenamiento new_var/Entrenamiento " + str(p) + ".txt"

        # with open(path, 'r') as f:
        #     nombres = f.readline().strip().split(' ')
        # f.close()
        # nombres = nombres[0:-1]
        # nombres.append('class')
>>>>>>> ef2ead75d86ed458cc05352d301dc95d1ccb2bf2

        nombres = ['Macho_id', 'Sigma_B', 'Sigma_B_comp', 'Eta_B', 'Eta_B_comp', 'stetson_L_B', 'stetson_L_B_comp', 'CuSum_B', 'CuSum_B_comp', 'B-R', 'B-R_comp', 'stetson_J', 'stetson_J_comp', 'stetson_K', 'stetson_K_comp', 'skew', 'skew_comp', 'kurt', 'kurt_comp', 'std', 'std_comp', 'beyond1_std', 'beyond1_std_comp', 'max_slope', 'max_slope_comp', 'amplitude', 'amplitude_comp', 'med_abs_dev', 'med_abs_dev_comp', 'class']

<<<<<<< HEAD
    # nombres = ['Macho_id', 'Sigma_B', 'Sigma_B_comp', 'Eta_B', 'Eta_B_comp', 'stetson_L_B', 'stetson_L_B_comp', 'CuSum_B', 'CuSum_B_comp', 'B-R', 'B-R_comp', 'stetson_J', 'stetson_J_comp', 'stetson_K', 'stetson_K_comp', 'skew', 'skew_comp', 'kurt', 'kurt_comp', 'std', 'std_comp', 'beyond1_std', 'beyond1_std_comp', 'max_slope', 'max_slope_comp', 'amplitude', 'amplitude_comp', 'med_abs_dev', 'med_abs_dev_comp', 'class']
    #
    # data = tree.pd.read_csv(path, sep=' ', header=None, names=nombres, skiprows=1, index_col=0)
    #
    # train = pd.DataFrame()
    # test = pd.DataFrame()
=======
        data = pd.read_csv(path, sep=' ', header=None, names=nombres, skiprows=1, index_col=0)

        # Hago cross validation
        skf = cross_validation.StratifiedKFold(data['class'], n_folds=2)

        results = []
        count = 1
        for train_index, test_index in skf:
            print 'Fold: ' + str(count)
            count += 1
            train, test = data.iloc[train_index], data.iloc[test_index]
>>>>>>> ef2ead75d86ed458cc05352d301dc95d1ccb2bf2

            clf = tree.Tree('confianza')
            clf.fit(train)

<<<<<<< HEAD
    # for i in data['class'].unique():
    #
    #     aux = data[data['class'] == i]
    #
    #     total = len(aux.index)
    #     fraccion = total / 10
    #
    #     train = train.append(aux.iloc[0:-fraccion])
    #     test = test.append(aux.iloc[-fraccion:])

    nombres = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

    data = pd.read_csv('iris_comp.data', header=None, names=nombres)

    clf = tree.Tree('gain')
    clf.fit(data)
    # clf = tree.Tree('confianza')
    # clf.fit(train)
    #
    # result = clf.predict_table(test)
    # matrix = clf.confusion_matrix(result)
    #
    #
    # # Serializo los resultados con pickle
    #
    # output = open( 'arbol 60.pkl', 'w')
    # pickle.dump(clf, output)
    # output.close()
    #
    # output = open( 'result 60.pkl', 'w')
    # pickle.dump(result, output)
    # output.close()
=======
            results.append(clf.predict_table(test))

        # train = pd.DataFrame()
        # test = pd.DataFrame()
        # # Genero un set de test con el 10% de los datos de cada clase

        # for i in data['class'].unique():

        #     aux = data[data['class'] == i]

        #     total = len(aux.index)
        #     fraccion = total / 10

        #     train = train.append(aux.iloc[0:-fraccion])
        #     test = test.append(aux.iloc[-fraccion:])
        # clf = tree.Tree('gain')
        # clf = tree.Tree('confianza')
        # clf.fit(train)
        # result = clf.predict_table(test)
        # matrix = clf.confusion_matrix(result)

        result = pd.concat(results)
        matrix = clf.confusion_matrix(result)

        # Serializo los resultados con pickle
        
        output = open( 'arbol ' + str(p) + '.pkl', 'w')
        pickle.dump(clf, output)
        output.close()

        output = open( 'result '+ str(p) + '.pkl', 'w')
        pickle.dump(result, output)
        output.close()
>>>>>>> ef2ead75d86ed458cc05352d301dc95d1ccb2bf2
