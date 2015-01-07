import pandas as pd
import numpy as np

from node import *
from CompNode import *
from UNode import *


class Tree:
    def __init__(self, criterium, max_depth=8, min_samples_split=10,
                 most_mass_threshold=0.9, min_mass_threshold=0.0127,
                 min_weight_threshold=0.0):
        self.root = []
        self.criterium = criterium
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.most_mass_threshold = most_mass_threshold
        self.min_mass_threshold = min_mass_threshold
        self.min_weight_threshold = min_weight_threshold

    # recibe un set de entrenamiento y ajusta el arbol
    def fit(self, data, y):
        if self.criterium == 'gain':
            self.root = Node(level=1, max_depth=self.max_depth,
                             min_samples_split=self.min_samples_split)
            self.root.fit(data, y)
        elif self.criterium == 'confianza':
            self.root = CompNode(data, level=1, max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split)
            self.root.fit(data)
        elif self.criterium == 'uncertainty':
            self.root = UNode(level=1, max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              most_mass_threshold=self.most_mass_threshold,
                              min_mass_threshold=self.min_mass_threshold)
            self.root.fit(data)

    def predict(self, tupla):
        """Returnes the predicted class of a tuple"""

        if self.criterium != 'uncertainty':
            return self.root.predict(tupla)
        else:
            # diccionario con las clases y sus probabilidades
            prediction = self.root.predict(tupla)

            # Busco la clase con la mayor probabilidad y su probabilidad
            maximo = max(prediction.values())
            clase = None
            for key in prediction.keys():
                if maximo == prediction[key]:
                    clase = key

            return clase, maximo

    def show(self):
        """Prints the tree structure"""
        self.root.show()

    # def predict_table(self, frame):
    #     """Returnes the original class, the prediction and its probability

    #     It serves as a testing mechanism of the performance of a classifier

    #     Parameters
    #     ----------
    #     frame: Dataframe of the data that must be classified. Each row is an
    #            object and each column is a feature
    #     """
    #     # Creo el frame e inserto la clase
    #     tabla = []
    #     for index, row in frame.iterrows():
    #         clase = row['class']
    #         predicted, confianza = self.predict(row)
    #         tabla.append([clase, predicted, confianza])

    #     return pd.DataFrame(tabla, index=frame.index,
    #                         columns=['original', 'predicted', 'trust'])

    def predict_table(self, frame, y):
        """Returnes the original class, the prediction and its probability

        It serves as a testing mechanism of the performance of a classifier

        Parameters
        ----------
        frame: Dataframe of the data that must be classified. Each row is an
               object and each column is a feature
        """
        # Creo el frame e inserto la clase
        tabla = []
        for index, row in frame.iterrows():
            clase = y[index]
            predicted, confianza = self.predict(row)
            tabla.append([clase, predicted, confianza])

        return pd.DataFrame(tabla, index=frame.index,
                            columns=['original', 'predicted', 'trust'])

    def confusion_matrix(self, table):
        """Generates a confusion matrix from the prediction table"""

        unique = np.unique(np.concatenate((table['original'].values,
                           table['predicted'].values), axis=1))

        matrix = np.zeros((len(unique), len(unique)))
        matrix = pd.DataFrame(matrix)
        matrix.columns = unique
        matrix.index = unique

        for index, row in table.iterrows():
            matrix[row[0]][row[1]] += row[2]

        return matrix

    def hard_matrix(self, table):
        """Generates a hard_confusion matrix for probabilistic classifiers"""

        unique = np.unique(np.concatenate((table['original'].values,
                           table['predicted'].values)))

        matrix = np.zeros((len(unique), len(unique)))
        matrix = pd.DataFrame(matrix)
        matrix.columns = unique
        matrix.index = unique

        for index, row in table.iterrows():
            matrix[row[0]][row[1]] += 1

        return matrix

    def precision(self, matrix, clase):
        """Shows the accuracy of a given class, based on a confusion matrix"""

        if clase in matrix.columns:
            correctos = matrix[clase].loc[clase]
            total = matrix[clase].sum()

            return correctos / total

        # A negative number is returned to show that there are no predictions
        # of the given class on the confusion matrix
        else:
            return -1

    def recall(self, matrix, clase):
        """Shows the recall of a given class, based on a confusion matrix"""

        if clase in matrix.columns:
            reconocidos = matrix[clase].loc[clase]
            total = matrix.loc[clase].sum()

            return reconocidos / total

        # A negative number is returned to show that there are no predictions
        # of the given class on the confusion matrix
        else:
            return -1

    def f_score(self, matrix, clase):
        """Shows the f_score of a given class, based on a confusion matrix"""

        acc = self.precision(matrix, clase)
        rec = self.recall(matrix, clase)

        # Neccesary check, in order to avoid divisions by zero
        if acc == 0 or rec == 0:
            return 0

        # Check that both are valid
        elif acc == -1 or rec == -1:
            return -1

        # Retorno f_score
        else:
            return 2 * acc * rec / (acc + rec)
