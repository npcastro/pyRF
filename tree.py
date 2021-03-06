# coding=utf-8

# Clase que engloba los distintos tipos de árboles de decisión
# -------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

from node import *
from UNode import *
from PNode import *
from FNode import *


class Tree:
    def __init__(self, criterium, max_depth=8, min_samples_split=10,
                 most_mass_threshold=0.9, min_mass_threshold=0.0127,
                 min_weight_threshold=0.0, parallel=None, n_jobs=1,
                 verbose=False):
        self.root = []
        self.criterium = criterium
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.most_mass_threshold = most_mass_threshold
        self.min_mass_threshold = min_mass_threshold
        self.min_weight_threshold = min_weight_threshold
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.feat_names = None
        self.verbose=verbose

    # recibe un set de entrenamiento y ajusta el arbol
    def fit(self, data, y):
        if self.criterium == 'gain':
            self.root = Node(level=1, max_depth=self.max_depth,
                             min_samples_split=self.min_samples_split, verbose=self.verbose)
            self.root.fit(data, y)

        elif self.criterium == 'uncertainty' and self.parallel is None:
            self.root = UNode(level=1, max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              most_mass_threshold=self.most_mass_threshold,
                              min_mass_threshold=self.min_mass_threshold,
                              verbose=self.verbose)
            data['class'] = y
            self.root.fit(data)

        elif self.criterium == 'uncertainty' and self.parallel == 'features':
            self.root = FNode(level=1, max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              most_mass_threshold=self.most_mass_threshold,
                              min_mass_threshold=self.min_mass_threshold, n_jobs=self.n_jobs,
                              verbose=self.verbose)
            data['class'] = y
            self.root.fit(data)

        elif self.criterium == 'uncertainty' and self.parallel == 'splits':
            self.root = PNode(level=1, max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              most_mass_threshold=self.most_mass_threshold,
                              min_mass_threshold=self.min_mass_threshold, n_jobs=self.n_jobs)
            data['class'] = y
            self.root.fit(data)

        self.feat_names = self.root.feat_names

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

    def predict_table(self, frame, y):
        """Returnes the original class, the prediction and its probability

        It serves as a testing mechanism of the performance of a classifier

        Parameters
        ----------
        frame: Dataframe of the data that must be classified. Each row is an
               object and each column is a feature
        y:     Real classes of the data that must be classified
        """
        # Creo el frame e inserto la clase
        tabla = []
        for index, row in frame.iterrows():
            clase = y[index]
            predicted, confianza = self.predict(row)
            tabla.append([clase, predicted, confianza])

        return pd.DataFrame(tabla, index=frame.index,
                            columns=['original', 'predicted', 'trust'])

    def get_splits(self):
        """Returns a dict with the positions of the splits made for each feature in the tree

        Parameters
        ----------
        return -> (dict) las llaves son los nombres de las feats presentes en el arbol
                  los valores son listas de floats que corresponden a los ptos de corte
        """
        splits = {}

        node_list = [self.root]

        while node_list:
            node = node_list.pop(0)

            if not node.is_leaf:
                splits.setdefault(node.feat_name, []).append(node.feat_value)
                node_list.append(node.right)
                node_list.append(node.left)

        return splits

    def get_split_counts(self, max_depth=float('inf')):
        """Retorna el numero de cortes que el árbol hace para cada feature. Según esto se puede
        realizar una medida de importancia para las features.

        Parameters
        ----------
        class: (string) nombre de la clase segun la cual se quiere filtrar. En este caso solo se 
                consideran las features que hacen cortes en los caminos para llegar a esa clase

        max_depth: (int) numero de niveles que se toman en consideración para hacer el conteo

        return -> (dict) las llaves son los nombres de las features que se ocuparon al entrenar el
                  árbol. Los valores son ints con los conteos
        """
        split_counts = {feat: 0 for feat in self.feat_names}

        node_list = [self.root]

        while node_list:
            node = node_list.pop(0)

            if not node.is_leaf:
                split_counts[node.feat_name] += 1
                if node.level < max_depth:
                    node_list.append(node.right)
                    node_list.append(node.left)

        return split_counts

    def get_feat_importance(self, criterion='entropy', max_depth=float('inf')):
        """Retorna la importancia relativa de cada una de las features con las que el árbol
        fue entrenado

        Parameters
        ----------

        criterion: (string) que tipo de medida se ocupa para determinar la importancia. Puede ser
              'gini', 'splits'
        """
        if criterion == 'entropy':
            importance_function = self.get_entropy_reduction
        elif criterion == 'splits':
            importance_function = self.get_split_counts
        

        scores = importance_function(max_depth)
        suma = float(sum(scores.values()))

        return { key: scores[key] / suma for key in scores.keys() }

    def get_entropy_reduction(self, max_depth=float('inf')):
        """Retorna la reducción total de entropía que cada feature aporta en el árbol.
        Esto se puede tomar como una medida de importancia para las features.

        Parameters
        ----------
        class: (string) nombre de la clase segun la cual se quiere filtrar. En este caso solo se 
                consideran las features que hacen cortes en los caminos para llegar a esa clase

        max_depth: (int) numero de niveles que se toman en consideración para hacer el conteo

        return -> (dict) las llaves son los nombres de las features que se ocuparon al entrenar el
                  árbol. Los valores son ints con los conteos
        """

        reduction_by_feat = {feat: 0.0 for feat in self.feat_names}

        node_list = [self.root]

        while node_list:
            node = node_list.pop(0)

            if not node.is_leaf:
                reduction = node.entropia - (node.left.mass * node.left.entropia +
                                             node.right.mass * node.right.entropia) / float(node.mass)

                reduction_by_feat[node.feat_name] += reduction
                if node.level < max_depth:
                    node_list.append(node.right)
                    node_list.append(node.left)

        return reduction_by_feat
