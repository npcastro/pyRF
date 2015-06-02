# coding=utf-8

import sys
import time
import math
import datetime
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

import pandas as pd

from node import *
import fnode_utils
import pyRF_prob


class FNode():
    def __init__(self, level=1, max_depth=8, min_samples_split=10, most_mass_threshold=0.9,
                 min_mass_threshold=0.0127, min_weight_threshold=0.01, n_jobs=1):
        """
        data (DataFrame): Each row represents an object, each column represents
            a feature. Must contain a column named 'class'
        level (int): The deepness level of the node
        max_depth (int): Max depth that the nodes can be splitted
        min_samples_split (int): Minimum number of tuples necesary for splitting
        most_mass_threshold (float): If a single class mass is over this threshold the node is
            considered a leaf
        min_mass_threshold (float):

        ESTO FALTA!!: If the total mass is below this threshold the node is no longer
            splitted.
        min_weight_threshold (float): Tuples with mass below this, are removed from the children.
            This value must be small or else, problem with probabilities may arise.
        """
        # Atributos particulares del nodo
        self.clase = ''
        self.feat_name = ""
        self.feat_value = None
        self.is_leaf = False
        self.is_left = False
        self.is_right = False
        self.left = None
        self.right = None
        self.level = level
        # self.split_type = split_type

        # Atributos generales del arbol
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.most_mass_threshold = most_mass_threshold
        self.min_mass_threshold = min_mass_threshold
        self.min_weight_threshold = min_weight_threshold
        self.n_jobs = n_jobs

    def add_left(self, left_data):
        self.left = self.__class__(self.level + 1, self.max_depth,
                                   self.min_samples_split, self.most_mass_threshold)
        self.left.fit(left_data)
        self.left.is_left = True

    def add_right(self, right_data):
        self.right = self.__class__(self.level + 1, self.max_depth,
                                    self.min_samples_split, self.most_mass_threshold)
        self.right.fit(right_data)
        self.right.is_right = True

    def check_leaf_condition(self):
        """ Determina se es necesario hacer un split de los datos
        """
        featuresfaltantes = self.filterfeatures()

        if self.data['class'].nunique() == 1 or len(featuresfaltantes) == 0:
            return False
        elif self.level >= self.max_depth:
            return False
        # Creo que esta condicion esta de mas. La de abajo ya lo abarca y mejor
        elif self.n_rows < self.min_samples_split:
            return False
        elif self.mass < self.min_samples_split:
            return False
        elif self.check_most_mass():
            return False
        else:
            return True

    def check_most_mass(self):
        """Check the most_mass_threshold condition"""

        mass_sum = self.data.groupby('class')['weight'].sum().to_dict()

        if max(mass_sum.values()) / self.mass >= self.most_mass_threshold:
            return True
        else:
            return False

    def filterfeatures(self):
        """Retorna las features a considerar en un nodo para hacer la particion"""
        filter_arr = []
        for f in self.data.columns:
            if ('_comp' not in f and '.l' not in f and '.r' not in f and '.std' not in f and
               f != 'weight' and f != 'class'):
                filter_arr.append(f.replace('.mean', ''))
        return filter_arr

    def fit(self, data):
        self.data = data
        self.entropia = fnode_utils.entropy(data.groupby('class')['weight'].sum().to_dict())
        self.mass = data['weight'].sum()
        self.n_rows = len(data.index)

        # Si es necesario particionar el nodo, llamo a split para hacerlo
        if self.check_leaf_condition():
            self.split()

            if self.feat_name != '':
                print 'Feature elegida: ' + self.feat_name
                print 'Pivote elegido: ' + str(self.feat_value)

                menores = self.get_menores(self.feat_name, self.feat_value)
                mayores = self.get_mayores(self.feat_name, self.feat_value)

                # There's a chance that the split that's been found leaves an empty dataframe
                # because non of the tuples has enough mass to be considerable
                if menores.empty or mayores.empty:
                    self.set_leaf()
                else:
                    self.add_right(mayores)
                    self.add_left(menores)

            else:
                self.set_leaf()

        # De lo contrario llamo a set_leaf para transformarlo en hoja
        else:
            self.set_leaf()

    def get_menores(self, feature_name, pivote):
        menores = []

        # limpio el nombre de la feature
        feature_name = feature_name.replace('.mean', '')
        menores = self.data[self.data[feature_name + '.l'] < pivote]

        menores = menores.apply(func=self.get_weight, axis=1, args=[pivote, feature_name, "menor"])
        menores = menores[menores["weight"] > self.min_weight_threshold]

        return pd.DataFrame(menores, index=menores.index)

    def get_mayores(self, feature_name, pivote):
        mayores = []

        # limpio el nombre de la feature
        feature_name = feature_name.replace('.mean', '')
        mayores = self.data[self.data[feature_name + '.r'] >= pivote]

        mayores = mayores.apply(func=self.get_weight, axis=1, args=[pivote, feature_name, "mayor"])
        mayores = mayores[mayores["weight"] > self.min_weight_threshold]

        return pd.DataFrame(mayores, index=mayores.index)

    def get_weight(self, tupla, pivote, feature_name, how):
        """ Determina la distribucion de probabilidad gaussiana acumulada entre dos bordes.

        pivote: valor de corte
        how: determina si la probabilidad se calcula desde l hasta pivote o desde pivote hasta r
        """

        left_bound = tupla[feature_name + '.l']
        right_bound = tupla[feature_name + '.r']

        if left_bound >= pivote and how == 'mayor' or right_bound <= pivote and how == 'menor':
            return tupla

        else:
            w = tupla['weight']
            mean = tupla[feature_name + '.mean']
            std = tupla[feature_name + '.std']

            feature_mass = pyRF_prob.cdf(pivote, mean, std, left_bound, right_bound)

            if how == 'menor':
                if (feature_mass >= self.min_mass_threshold):
                    tupla['weight'] = min(w * feature_mass, 1)
                else:
                    tupla['weight'] = 0
                # tupla[feature_name+'.r'] = min(pivote, tupla[feature_name + '.r'])
                tupla[feature_name + '.r'] = pivote
                return tupla

            elif how == 'mayor':
                feature_mass = 1 - feature_mass
                if (feature_mass >= self.min_mass_threshold):
                    tupla['weight'] = min(w * feature_mass, 1)
                else:
                    tupla['weight'] = 0
                # tupla[feature_name+'.l'] = max(pivote, tupla[feature_name + '.l'])
                tupla[feature_name + '.l'] = pivote
                return tupla

    def predict(self, tupla, prediction={}, w=1):
        # Si es que es el nodo raiz
        if len(prediction.keys()) == 0:
            prediction = {c: 0.0 for c in self.data['class'].unique()}

        if self.is_leaf:
            aux = deepcopy(prediction)
            aux[self.clase] += w
            return aux

        # Puede que falte chequear casos bordes, al igual que lo hago en get_menores y get_mayores
        else:
            feature_name = self.feat_name.replace('.mean', '')
            mean = tupla[feature_name + '.mean']
            std = tupla[feature_name + '.std']
            l = tupla[feature_name + '.l']
            r = tupla[feature_name + '.r']
            pivote = self.feat_value

            w_left = min(w * pyRF_prob.cdf(pivote, mean, std, l, r), 1)
            w_right = min(w * (1 - pyRF_prob.cdf(pivote, mean, std, l, r)), 1)

            a = self.right.predict(tupla, prediction, w_right)
            b = self.left.predict(tupla, prediction, w_left)

            # Tengo que retornar la suma elementwise de los diccionarios a y b
            return {key: a[key] + b[key] for key in a}

    # Convierte el nodo en hoja. Colocando la clase mas probable como resultado
    def set_leaf(self):
        self.is_leaf = True
        try:
            self.clase = self.data.groupby('class')['weight'].sum().idxmax()
        except Exception as inst:
            print self.data['class'].tolist()
            print self.data['weight'].tolist()
            print inst           # __str__ allows args to be printed directly
            x, y = inst.args
            print 'x =', x
            print 'y =', y
            raise

    def show(self, linea=""):
        if self.is_leaf:
            print linea + '|---- ' + str(self.clase)

        elif self.is_left:
            self.right.show(linea + '|     ')
            print linea + '|- ' + self.feat_name + ' ' + '(' + ("%.2f" % self.feat_value) + ')'
            self.left.show(linea + '      ')

        elif self.is_right:
            self.right.show(linea + '      ')
            print linea + '|- ' + self.feat_name + ' ' + '(' + ("%.2f" % self.feat_value) + ')'
            self.left.show(linea + '|     ')

        # Es el nodo raiz
        else:
            self.right.show(linea + '      ')
            print linea + '|- ' + self.feat_name + ' ' + '(' + ("%.2f" % self.feat_value) + ')'
            self.left.show(linea + '      ')

    def get_relevant_columns(self, feat_name, data):
        """Returns all the columns corresponding to the selected features plus 
        the weight and the class

        Parameters
        ----------
        feat_name: Name of the feature.
        data: Dataframe where to look for the columns
        """
        
        columnas = data.columns.tolist()
        indices = [i for i, s in enumerate(columnas) if feat_name in s]

        filtered_cols = data.columns[indices].tolist() + ['class', 'weight']

        return data[filtered_cols]

    def split(self):
        """Searches the best possible split for the node.

        After it finishes, it sets self.feat_name and self.feat_value
        """

        print '\n ################ \n'
        print 'Profundidad del nodo: ' + str(self.level)
        print 'Numero de tuplas en nodo: ' + str(self.n_rows)
        print 'Masa total del nodo: ' + str(self.mass)
        print '\n ################ \n'

        # Inicializo la ganancia de info en el peor nivel posible
        max_gain = 0

        # Obtengo los nombres de las features a probar
        candidate_features = self.filterfeatures()

        start_time = time.time()

        filtered_data = [self.get_relevant_columns(feat_name, self.data) for feat_name in candidate_features]

        partial_eval = partial(fnode_utils.eval_feature, entropia=self.entropia, mass = self.mass)
        pool = Pool(processes=self.n_jobs)
        clip = lambda a, b: b if a < b else a / b
        chunks = clip(len(candidate_features), abs(self.n_jobs))

        # First map applies function to all candidate features
        gains_pivots_tuples = pool.map(partial_eval, zip(candidate_features, filtered_data), chunks)
        pool.close()
        pool.join()

        # Second map unzips the values into two different lists
        gains, pivots = map(list, zip(*gains_pivots_tuples))

        for i, gain in enumerate(gains):
            if gain > max_gain:
                max_gain = gain
                self.feat_value = pivots[i]
                self.feat_name = candidate_features[i]

        end_time = time.time()
        print 'Tiempo tomado por nodo: ' + str(datetime.timedelta(0, end_time - start_time))
