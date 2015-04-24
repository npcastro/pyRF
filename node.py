from __future__ import division
from collections import Counter
import math
import sys

import numpy as np
from scipy import stats

# Data es un dataframe que tiene que contener una columna class.
# La cual el arbol intenta predecir.
# podria pensar en relajar esto y simplemente indicar cual es la variable
#a predecir.


class Node:
    """Represents the internal and final nodes of a decision tree"""

    def __init__(self, level=1, max_depth=8, min_samples_split=10):
        """
        data (DataFrame): Each row represents an object, each column represents
                          a feature. Must contain a column named 'class'
        level (int): The deepness level of the node
        max_depth (int): Max depth that the nodes can be splitted
        min_samples_split (int): Minimum number of tuples necesary for splitting
        """

        # Atributos particulares del nodo
        self.is_leaf = False
        self.clase = ''
        self.feat_name = ""
        self.feat_value = None
        self.left = None
        self.right = None
        self.is_left = False
        self.is_right = False
        self.level = level

        # Atributos generales del arbol
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def get_class_distribution(self, y):
        """Produces a dictionary with the amount of tuples of each class
        """
        dist = {}

        for label in y:
            if label in dist.keys():
                dist[label] += 1
            else:
                dist[label] = 1

        return dist

    def fit(self, data, y):
        self.data = data
        self.y = np.array(y)
        # self.entropia = self.entropy(data.groupby('class').size().to_dict())
        self.entropia = self.entropy(self.get_class_distribution(y))
        self.n_rows = len(y)

        # Si es necesario particionar el nodo, llamo a split para hacerlo
        if self.check_leaf_condition():
            self.split()

            if self.feat_name != '':
                print '\n'
                print 'Feature elegida: ' + self.feat_name
                print 'Pivote elegido: ' + str(self.feat_value)

                # menores = self.get_menores(self.feat_name, self.feat_value)
                # mayores = self.get_mayores(self.feat_name, self.feat_value)

                # self.add_right(mayores)
                # self.add_left(menores)

                menores_X, menores_y = self.get_menores(self.feat_name, self.feat_value)
                mayores_X, mayores_y = self.get_mayores(self.feat_name, self.feat_value)

                self.add_left(menores_X, menores_y)
                self.add_right(mayores_X, mayores_y)

            else:
                self.set_leaf()

        # De lo contrario llamo a set_leaf para transformarlo en hoja
        else:
            self.set_leaf()

    # Busca el mejor corte posible para el nodo
    def split(self):

        # Inicializo la ganancia de info en el peor nivel posible
        max_gain = 0

        #Obtengo los nombres de las features a probar
        filterfeatures = self.filterfeatures()

        print '\n ################ \n'
        print 'Profundidad del nodo: ' + str(self.level)
        print 'Numero de tuplas en nodo: ' + str(self.n_rows)
        print '\n ################ \n'

        for feature in filterfeatures:

            sys.stdout.write("\r\x1b[K" + 'Evaluando feature: ' + feature)
            sys.stdout.flush()

            # # Ordeno el frame segun la feature indicada
            # data_por_media = self.data.sort(feature, inplace=False)

            # #Transformo la informacion relevante de esta feature a listas
            # mean_list = data_por_media[feature].tolist()
            # class_list = data_por_media['class'].tolist()

            # Ordeno los datos segun la feature que se esta probando
            sort_index = np.argsort(self.data[feature].tolist())
            mean_list = self.data[feature].iloc[sort_index].tolist()
            class_list = self.y[sort_index]

            unique_means = list(set(mean_list))
            unique_means.sort()

            # Creo diccionarios para guardar la masa de los estrictos menores y estrictos mayores,
            # y asi no calcularla continuamente.
            # Los menores parten vacios y los mayores parten con toda la masa
            clases = list(set(class_list))
            menores = {c: 0.0 for c in clases}
            mayores = dict(Counter(class_list))

            index = 0
            old_index = 0

            # Me muevo a traves de los posibles pivotes.
            if len(unique_means) > 1:
                for pivote in unique_means:
                    index = self.update_index(pivote, index, mean_list)

                    for i in xrange(old_index, index):
                        menores[class_list[i]] += 1
                        mayores[class_list[i]] -= 1
                    old_index = index

                    # Calculo la ganancia de informacion para esta variable
                    pivot_gain = self.gain(menores, mayores)

                    if pivot_gain > max_gain:
                        max_gain = pivot_gain
                        self.feat_value = pivote
                        self.feat_name = feature

    def update_index(self, pivote, index, mean_list):
        while mean_list[index] < pivote:
            index += 1
        return index

    # def get_menores(self, feature, pivote):
    #     return self.data[self.data[feature] < pivote]

    # def get_mayores(self, feature, pivote):
    #     return self.data[self.data[feature] >= pivote]

    def get_menores(self, feature, pivote):
        aux = np.array(self.data[feature] < pivote)
        return self.data[aux], self.y[aux]

    def get_mayores(self, feature, pivote):
        aux = np.array(self.data[feature] >= pivote)
        return self.data[aux], self.y[aux]

    # Retorna las features a considerar en un nodo para hacer la particion
    def filterfeatures(self):
        filter_arr = []
        for f in self.data.columns:
            if (not '_comp' in f and not '.l' in f and not '.r' in f and not '.std' in f and
               f != 'weight' and f != 'class'):
                filter_arr.append(f)
        return filter_arr

    # determina se es necesario hacer un split de los datos
    def check_leaf_condition(self):
        featuresfaltantes = self.filterfeatures()

        if len(np.unique(self.y)) == 1 or len(featuresfaltantes) == 0:
            return False
        elif self.level >= self.max_depth:
            return False
        elif self.n_rows < self.min_samples_split:
            return False
        else:
            return True

    # retorna una lista con los todos los threshold a evaluar para buscar la mejor separacion
    def get_pivotes(self, feature, calidad='exact'):

        if calidad == 'exact':
            return feature[1:].unique()
        elif calidad == 'aprox':
            minimo = feature.min()
            maximo = feature.max()
            step = maximo - minimo / 100
            pivotes = []
            for i in xrange(100):
                pivotes.append(minimo + step * i)

            return pivotes

    # Convierte el nodo en hoja. Colocando la clase mas probable como resultado
    def set_leaf(self):
        self.is_leaf = True
        self.clase = stats.mode(self.y)[0][0]
        # self.clase = self.data['class'].value_counts().idxmax()

    def add_left(self, left_data, y):
        self.left = self.__class__(self.level + 1, self.max_depth, self.min_samples_split)
        self.left.fit(left_data, y)
        self.left.is_left = True

    def add_right(self, right_data, y):
        self.right = self.__class__(self.level + 1, self.max_depth, self.min_samples_split)
        self.right.fit(right_data, y)
        self.right.is_right = True

    def predict(self, tupla, confianza=1):
        if self.is_leaf:
            return self.clase, confianza
        else:
            if tupla[self.feat_name] < self.feat_value:
                return self.left.predict(tupla)
            else:
                return self.right.predict(tupla)

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

    # Retorna la ganancia de dividir los datos en menores y mayores.
    def gain(self, menores, mayores):
        gain = (self.entropia - (sum(menores.values()) * self.entropy(menores) +
                sum(mayores.values()) * self.entropy(mayores)) / self.n_rows)
        return gain

    def entropy(self, data):
        """Retorna la entropia de un grupo de datos.

        data: diccionario donde las llaves son nombres de clases y los valores
              sumas (o conteos) de valores.
        """

        total = sum(data.values())
        entropia = 0

        for clase in data.keys():
            if data[clase] != 0:
                entropia -= (float(data[clase]) / total) * np.log2(float(data[clase]) / total)
        return entropia
