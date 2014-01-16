import pandas as pd
import numpy as np

from node import *


class Tree:
    def __init__(self, criterium):
        self.root = []
        self.criterium = criterium

    # recibe un set de entrenamiento y ajusta el arbol
    def fit(self, data):
        self.root = Node(data, self.criterium)

    # recibe un dato y retorna prediccion
    def predict(self, tupla):
        return self.root.predict(tupla)

    # seria bueno poder ver la estructura del arbol.
    def show(self):
        self.root.show()

    # recibe un frame completo y retorna otro frame con la clase original, la predicha y la confianza de la prediccion
    def predict_table(self, frame):
        # Creo el frame e inserto la clase
        tabla = []
        for index, row in frame.iterrows():
            clase = row['class']
            predicted, confianza = self.root.predict(row)
            tabla.append([clase, predicted, confianza])

        return pd.DataFrame(tabla, index=frame.index, columns=['original', 'predicted', 'trust'])

    #Matriz de confusion a partir de tabla de prediccion
    def confusion_matrix(self, table):

        unique = np.unique(np.concatenate((table['original'].values, table['predicted'].values), axis=1))

        matrix = np.zeros((len(unique), len(unique)))
        matrix = pd.DataFrame(matrix)
        matrix.columns = unique
        matrix.index = unique

        for index, row in table.iterrows():
            matrix[row[0]][row[1]] += row[2]

        return matrix

    # Matriz de confusion hard a partir de la tabla de prediccion
    def hard_matrix(self, table):
        unique = np.unique(np.concatenate((table['original'].values, table['predicted'].values), axis=1))        

        matrix = np.zeros((len(unique), len(unique)))
        matrix = pd.DataFrame(matrix)
        matrix.columns = unique
        matrix.index = unique

        for index, row in table.iterrows():
            matrix[row[0]][row[1]] += 1

        return matrix

    # Retorna el accuracy para una clase en particular
    def accuracy(self, matrix, clase):
        
        if clase in matrix.columns:
            correctos = matrix[clase].loc[clase]
            total = matrix[clase].sum()

            return correctos / total

        # Retorno un numero negativo para identificar que no hay predicciones de esa clase la matriz de confusion
        else:
            return -1


    # Retorna el recall para una clase en particular
    def recall(self, matrix, clase):
        
        if clase in matrix.columns:
            reconocidos = matrix[clase].loc[clase]
            total = matrix.loc[clase].sum()

            return reconocidos / total

        # Retorno un numero negativo para identificar que no hay predicciones de esa clase la matriz de confusion
        else:
            return -1

    def f_score(self, matrix, clase):
        acc = self.accuracy(matrix, clase)
        rec = self.recall(matrix, clase)

        # Reviso esto, para evitar divisiones por cero
        if acc == 0 or rec == 0:
            return 0

        # Reviso que ambos sean validos
        elif acc == -1 or rec == -1:
            print 'No hay predicciones para esa clase en la matriz de confusión'

        # Retorno f_score
        else:
            return 2 * acc * rec / (acc + rec)
