import tree
import pandas as pd
import numpy as np

if __name__ == '__main__':

    # path = "/Users/npcastro/workspace/Features/Resultados 20.txt"
    path = "/Users/npcastro/workspace/Features/Resultados 40.txt"
    # path = "/Users/npcastro/workspace/Features/Resultados 60.txt"
    # path = "/Users/npcastro/workspace/Features/Resultados 80.txt"

    # Por alguna razon no puedo leer del header de los archivos la palabra class
    # with open(path, 'r') as f:
    #     nombres = f.readline().strip().split(' ')
    # f.close()
    # nombres = nombres[0:-1]
    # nombres.append('class')

    nombres = ['Macho_id', 'Sigma_B', 'Sigma_B_comp', 'Eta_B', 'Eta_B_comp', 'stetson_L_B', 'stetson_L_B_comp', 'CuSum_B', 'CuSum_B_comp', 'B-R', 'B-R_comp', 'stetson_J', 'stetson_J_comp', 'stetson_K', 'stetson_K_comp', 'skew', 'skew_comp', 'kurt', 'kurt_comp', 'std', 'std_comp', 'beyond1_std', 'beyond1_std_comp', 'max_slope', 'max_slope_comp', 'amplitude', 'amplitude_comp', 'med_abs_dev', 'med_abs_dev_comp', 'class']

    data = tree.pd.read_csv(path, sep=' ', header=None, names=nombres, skiprows=1, index_col=0)

    train = pd.DataFrame()
    test = pd.DataFrame()

    # Genero un set de test con el 10% de los datos de cada clase

    # for i in range(2,10):
    for i in data['class'].unique():

        aux = data[data['class'] == i]

        total = len(aux.index)
        fraccion = total / 10

        train = train.append(aux.iloc[0:-fraccion])
        test = test.append(aux.iloc[-fraccion:])

    # clf = Tree('gain')
    clf = tree.Tree('confianza')
    clf.fit(train)

    result = clf.predict_table(test)
    matrix = clf.confusion_matrix(result)
