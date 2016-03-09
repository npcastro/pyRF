# Voy a hacer graficos con la distribucion  de los datos en cada variable
# y los cortes que hace el arbol de decision

# path = 'sets/Macho shaken/Macho shake 5.csv'
# path = 'sets/Macho random/Macho random 5.csv'

import matplotlib.pyplot as plt
import pandas as pd
import pickle

split_dict = pickle.load(open('Macho splits.pkl', 'rb'))
path = 'sets/Macho.csv'

data = pd.read_csv(path, index_col=0)
clases = data['class'].unique().tolist()
colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'brown']

result_dir = 'Resultados/Comparacion/Graficos/cortes/'

cols = data.columns.tolist()[:-2]

for col in cols:
    if col in split_dict:
        aux = data[[col, 'class']]

        plt.figure()

        for clase, color in zip(clases, colors):
            puntos = aux[aux['class'] == clase]
            puntos = puntos[col]

            plt.scatter(puntos, [0]*len(puntos), marker='+', color=color, s=20*2**2)


        for l in split_dict[col]:
            plt.axvline(l)

        # plt.show()
        plt.savefig(result_dir + str(col) + '.png')
        plt.close()

