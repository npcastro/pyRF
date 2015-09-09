# coding=utf-8

import sys
import time
import math
import datetime
from copy import deepcopy

import pandas as pd

from node import *
import pyRF_prob


class UNode():
    def __init__(self, level=1, max_depth=8, min_samples_split=10, most_mass_threshold=0.9,
                 min_mass_threshold=0.0127, min_weight_threshold=0.01):
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

    def check_unique_presence(self, values):
        aux = set(values)

        if 0 in aux and len(aux) == 2:
            return True
        else:
            return False

    def entropy(self, data):
        """Calculates the entropy of a group of data
        data: dicctionary where the keys are class names, and the values are counts or sums of mass
        """

        total = float(sum(data.values()))
        entropia = 0

        for clase in data.keys():
            if data[clase] != 0:
                entropia -= (data[clase] / total) * np.log2(data[clase] / total)

        return entropia

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
        self.entropia = self.entropy(data.groupby('class')['weight'].sum().to_dict())
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

    def fix_numeric_errors(self, num_dict):
        """Masses that are extremely small are rounded to zero."""

        for key in num_dict.keys():
            if abs(num_dict[key]) < 1e-10 and num_dict[key] < 0:
                num_dict[key] = 0

        return num_dict

    def gain(self, menores, mayores):
        """Retorna la ganancia de dividir los datos en menores y mayores
        Menores y mayores son diccionarios donde la llave es el nombre
        de la clase y los valores son la suma de masa para ella.
        """
        gain = (self.entropia - (sum(menores.values()) * self.entropy(menores) +
                sum(mayores.values()) * self.entropy(mayores)) / self.mass)

        return gain

    def get_menores(self, feature_name, pivote):
        menores = []

        menores = self.data[self.data[feature_name + '.l'] < pivote]

        menores = menores.apply(func=self.get_weight, axis=1, args=[pivote, feature_name, "menor"])
        menores = menores[menores["weight"] > self.min_weight_threshold]

        return pd.DataFrame(menores, index=menores.index)

    def get_mayores(self, feature_name, pivote):
        mayores = []

        mayores = self.data[self.data[feature_name + '.r'] >= pivote]

        mayores = mayores.apply(func=self.get_weight, axis=1, args=[pivote, feature_name, "mayor"])
        mayores = mayores[mayores["weight"] > self.min_weight_threshold]

        return pd.DataFrame(mayores, index=mayores.index)

    def get_split_candidates(self, data, feature_name, split_type='simple'):
        """Returns a list of all the points of a feature that must be tested as a split point
        """
        if split_type == 'simple':
            bounds = (data[feature_name + '.l'].tolist() +
                      data[feature_name + '.r'].tolist())

            print 'Simple ' + str(len(np.unique(bounds)))
            return np.unique(bounds)

        else:
            bounds = self.get_class_changes(data[feature_name + '.l'].tolist(),
                                            data[feature_name + '.r'].tolist(),
                                            data['class'].tolist())
            bounds = np.unique(bounds)
            print 'Nuevo ' + str(len(bounds))
            return bounds

    # Parece que estoy guardando la clase actual por las puras
    def get_class_changes(self, left_values, right_values, clases):
        presence = {c: 0 for c in set(clases)}
        bounds = []

        left_index = 1
        right_index = 0

        # I add the values for the first point (neccesarily a left bound)
        clase_actual = clases[0]
        presence[clase_actual] = 1

        while right_index < len(right_values):

            if left_index < len(left_values) and \
               left_values[left_index] <= right_values[right_index]:
                value = left_values[left_index]
                clase_actual = clases[left_index]
                presence[clase_actual] += 1

                left_index += 1

                right = False

            else:
                value = right_values[right_index]
                clase_actual = clases[right_index]
                # presence[clase_actual] -= 1

                right_index += 1
                right = True

            # There's no one. I have to check the next border
            if len(np.unique(presence.values())) == 1 and 0 in presence.values():
                if right_index < len(right_values) - 1:
                    if clases[right_index + 1] != clase_actual:
                        bounds.append(value)
                else:
                    continue

            # There's one class with presence, all the other have zeroes
            elif self.check_unique_presence(presence.values()):
                continue

            else:
                bounds.append(value)

            if right:
                presence[clase_actual] -= 1

        return bounds

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

    def update_indexes(self, menores_index, mayores_index, pivote, limites_l, limites_r):
        """Updates the strictly inferior and superior tuples and updates to the new pivot.
        Parameters
        ----------
        menores_index: The index of the strictly inferior data to the last pivot
        mayores_index: The index of the strictly superior data to the last pivot
        pivote: The new pivot that splits the data in two
        limites_l: The left margin of the distributions of the data
        limites_r: The right margin of the distributions of the data
        """

        ultimo_r_menor = limites_r[menores_index]

        # Itero hasta encontrar una tupla que NO sea completamente menor que el pivote
        while(ultimo_r_menor < pivote and menores_index < len(limites_r) - 1):
            menores_index += 1
            ultimo_r_menor = limites_r[menores_index]

        ultimo_l_mayor = limites_l[mayores_index]

        # Itero hasta encontrar una tupla que SEA completamente mayor que el pivote
        while(ultimo_l_mayor < pivote and mayores_index < len(limites_l) - 1):
            ultimo_l_mayor = limites_l[mayores_index]
            mayores_index += 1

        return menores_index, mayores_index

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
        filterfeatures = self.filterfeatures()

        start_time = time.time()
        for f in filterfeatures:

            sys.stdout.write("\r\x1b[K" + 'Evaluando feature: ' + f)
            sys.stdout.flush()

            # Ordeno el frame segun la media de la variable
            data_por_media = self.data.sort(f + '.mean', inplace=False)

            # Transformo la informacion relevante de esta feature a listas
            w_list = data_por_media['weight'].tolist()
            mean_list = data_por_media[f + '.mean'].tolist()
            std_list = data_por_media[f + '.std'].tolist()
            left_bound_list = data_por_media[f + '.l'].tolist()
            right_bound_list = data_por_media[f + '.r'].tolist()
            class_list = data_por_media['class'].tolist()

            menores_index = 0
            mayores_index = 0

            old_menores_index = 0
            old_mayores_index = 0

            # Obtengo las clases existentes
            clases = list(set(class_list))

            # Creo diccionarios para guardar la masa de los estrictos menores y estrictos mayores,
            # y asi no calcularla continuamente.
            # Los menores parten vacios y los mayores parten con toda la masa
            menores_estrictos_mass = {c: 0.0 for c in clases}
            mayores_estrictos_mass = data_por_media.groupby('class')['weight'].sum().to_dict()

            # Me muevo a traves de los posibles pivotes
            # for pivote in self.get_split_candidates(feature_name, split_type=self.split_type):
            # for pivote in self.get_split_candidates(feature_name):
            for pivote in self.get_split_candidates(data_por_media, f, split_type='otro'):

                # Actualizo los indices
                menores_index, mayores_index = self.update_indexes(
                    menores_index, mayores_index,
                    pivote, left_bound_list, right_bound_list
                )

                # Actualizo la masa estrictamente menor y mayor
                for j in xrange(old_menores_index, menores_index):
                    menores_estrictos_mass[class_list[j]] += w_list[j]

                for j in xrange(old_mayores_index, mayores_index):
                    mayores_estrictos_mass[class_list[j]] -= w_list[j]

                # Actualizo los indices anteriores
                old_menores_index, old_mayores_index = menores_index, mayores_index

                # Guardo las listas de elementos afectados por el pivote actual
                w_list_afectada = w_list[menores_index:mayores_index]
                mean_list_afectada = mean_list[menores_index:mayores_index]
                std_list_afectada = std_list[menores_index:mayores_index]
                left_bound_list_afectada = left_bound_list[menores_index:mayores_index]
                right_bound_list_afectada = right_bound_list[menores_index:mayores_index]
                class_list_afectada = class_list[menores_index:mayores_index]

                menores, mayores = self.split_tuples_by_pivot(
                    w_list_afectada, mean_list_afectada, std_list_afectada,
                    left_bound_list_afectada, right_bound_list_afectada, class_list_afectada,
                    pivote, deepcopy(menores_estrictos_mass), deepcopy(mayores_estrictos_mass)
                )

                if not any(menores) or not any(mayores):
                    continue

                elif sum(menores.values()) == 0 or sum(mayores.values()) == 0:
                    continue

                # Calculo la ganancia de informacion para esta variable
                menores = self.fix_numeric_errors(menores)
                mayores = self.fix_numeric_errors(mayores)
                pivot_gain = self.gain(menores, mayores)

                # En caso de error
                # if math.isnan(pivot_gain):
                #     aux = data_por_media.groupby('class')['weight'].sum().to_dict()
                #     print 'Total: ' + str(aux)
                #     print 'Menores: ' + str(menores)
                #     print 'Mayores: ' + str(mayores)
                #     print 'Menores_estrictos_mass: ' + str(menores_estrictos_mass)
                #     print 'Mayores_estrictos_mass: ' + str(mayores_estrictos_mass)
                #     print 'Menores_index: ' + str(menores_index)
                #     print 'Mayores_index: ' + str(mayores_index)
                #     sys.exit("Ganancia de informacion indefinida")

                if pivot_gain > max_gain:
                    max_gain = pivot_gain
                    self.feat_value = pivote
                    self.feat_name = f

            # Para profiling de codigo
            # break

        end_time = time.time()
        print 'Tiempo tomado por nodo: ' + str(datetime.timedelta(0, end_time - start_time))

    def split_tuples_by_pivot(self, w_list, mean_list, std_list, left_bound_list, right_bound_list,
                              class_list, pivote, menores, mayores):
        """divides a group of data according to a pivot
        It operates along all the data. And then returns two dictionaries with the total sum
        of the mass separated by class.
        Returns:
            menores: Dictionary for the data thats inferior than the pivot
            mayores: Dictionary for the data thats superior to the pivot
        """
        clip = lambda x, l, r: l if x < l else r if x > r else x

        # Este loop es fundamental paralelizarlo
        for i in xrange(len(class_list)):
            cum_prob = pyRF_prob.cdf(pivote, mean_list[i], std_list[i], left_bound_list[i],
                                     right_bound_list[i])

            cum_prob = clip(cum_prob, 0, 1)

            # En vez de agregar estas cantidades hago un submetodo que las retorne
            # Hago un map y dsp las unzipeo y las sumo según su clase
            menores[class_list[i]] += w_list[i] * cum_prob
            mayores[class_list[i]] += w_list[i] * (1 - cum_prob)

        return menores, mayores
