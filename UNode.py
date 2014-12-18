import sys
import time
import math
import datetime
from copy import deepcopy

import pandas as pd

from node import *
import pyRF_prob


class UNode(Node):
    def __init__(self, level=1, max_depth=8, min_samples_split=10, most_mass_threshold=0.9,
                 min_mass_threshold=0.0127, min_weight_threshold=0.0):
        """
        data (DataFrame): Each row represents an object, each column represents
            a feature. Must contain a column named 'class'
        level (int): The deepness level of the node
        max_depth (int): Max depth that the nodes can be splitted
        min_samples_split (int): Minimum number of tuples necesary for splitting
        most_mass_threshold (float): If a single class mass is over this threshold the node is
            considered a leaf
        min_mass_threshold (float): If the total mass is below this threshold the node is no longer
            splitted
        min_weight_threshold (float): Tuples with mass below this, are removed from the children.
            This value must be small or else, problem with probabilities
            may arise
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
        #Creo que esta condicion esta de mas. La de abajo ya lo abarca y mejor
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

        for clase in mass_sum.keys():
            if mass_sum[clase] / self.mass >= self.most_mass_threshold:
                return True

        return False

    def entropy(self, data):
        """ Retorna la entropia de un grupo de datos.
        data: diccionario donde las llaves son nombres de clases y los valores sumas
            (o conteos de valores)
        """

        total = sum(data.values())
        entropia = 0

        for clase in data.keys():
            if data[clase] != 0:
                entropia -= (data[clase] / total) * np.log2(data[clase] / total)

        return entropia

    def filterfeatures(self):
        """Retorna las features a considerar en un nodo para hacer la particion"""
        filter_arr = []
        for f in self.data.columns:
            if (not '_comp' in f and not '.l' in f and not '.r' in f and not '.std' in f and
               f != 'weight' and f != 'class'):
                filter_arr.append(f)
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

        # limpio el nombre de la feature
        feature_name = feature_name.replace('.mean', '')
        menores = self.data[self.data[feature_name + '.l'] < pivote]

        menores = menores.apply(func=self.get_weight, axis=1, args=[pivote, feature_name, "menor"])
        #menores = menores[menores["weight"] > self.min_weight_threshold]

        return pd.DataFrame(menores, index=menores.index)

    def get_mayores(self, feature_name, pivote):
        mayores = []

        # limpio el nombre de la feature
        feature_name = feature_name.replace('.mean', '')
        mayores = self.data[self.data[feature_name + '.r'] >= pivote]

        mayores = mayores.apply(func=self.get_weight, axis=1, args=[pivote, feature_name, "mayor"])
        #mayores = mayores[mayores["weight"] > self.min_weight_threshold]

        return pd.DataFrame(mayores, index=mayores.index)

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

    def get_relevant_columns(data, feature_name, menores_index=0, mayores_index=0):
        """Returns the relevant information of a dataframe as lists"""
        w_list = data['weight'].tolist()
        mean_list = data[feature_name + '.mean'].tolist()
        std_list = data[feature_name + '.std'].tolist()
        left_bound_list = data[feature_name + '.l'].tolist()
        right_bound_list = data[feature_name + '.r'].tolist()
        class_list = data['class'].tolist()

        return w_list, mean_list, std_list, left_bound_list, right_bound_list, class_list

    def get_split_candidates(self, feature_name):
        """ Retorna todos los valores segun los que se debe intentar cortar una feature
        """
        bounds = (self.data[feature_name + '.l'].unique().tolist() +
                  self.data[feature_name + '.r'].unique().tolist())

        bounds.sort()
        #print 'Numero de candidatos: ' + str(len(bounds))
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

            # w_left = self.get_weight(w, mean, std, l, r, pivote, 'menor')
            # w_right = self.get_weight(w, mean, std, l, r, pivote, 'mayor')

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

    # Busca el mejor corte posible para el nodo
    def split(self):

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

            # Output que se sobreescribe
            sys.stdout.write('Evaluando feature: ' + f)
            sys.stdout.flush()
            sys.stdout.write('\r')
            sys.stdout.flush()

            # Limpio el nombre de la feature
            feature_name = f.replace('.mean', '')

            # Ordeno el frame segun la media de la variable
            data_por_media = self.data.sort(f, inplace=False)

            # print '\n'
            #Transformo la informacion relevante de esta feature a listas
            w_list = data_por_media['weight'].tolist()
            mean_list = data_por_media[feature_name + '.mean'].tolist()
            std_list = data_por_media[feature_name + '.std'].tolist()
            left_bound_list = data_por_media[feature_name + '.l'].tolist()
            right_bound_list = data_por_media[feature_name + '.r'].tolist()
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
            for pivote in self.get_split_candidates(feature_name):
            # for pivote in mean_list:

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
                    # print 'gain elegido: ' + str(pivot_gain)
                    # print 'pivote elegido: ' + str(pivote)

                    max_gain = pivot_gain
                    self.feat_value = pivote
                    self.feat_name = feature_name + '.mean'

        end_time = time.time()
        print 'Tiempo tomado por nodo: ' + str(datetime.timedelta(0, end_time - start_time))
            # break # Para testear cuanto se demora una sola feature

    def split_tuples_by_pivot(self, w_list, mean_list, std_list, left_bound_list, right_bound_list,
                              class_list, pivote, menores, mayores):
        """Splits a group of data and divides it according to a pivot

        It operates along all the data. And then returns two dictionaries with the total sum
        of the mass separated by class.

        Returns:
            menores: Dictionary for the data thats inferior than the pivot
            mayores: Dictionray for the data thats superior to the pivot
        """
        clip = lambda x, l, r: l if x < l else r if x > r else x

        for i in xrange(len(class_list)):
            cum_prob = pyRF_prob.cdf(pivote, mean_list[i], std_list[i], left_bound_list[i],
                                     right_bound_list[i])

            # if cum_prob > 1 or cum_prob < 0:
            #     print cum_prob

            cum_prob = clip(cum_prob, 0, 1)

            menores[class_list[i]] += w_list[i] * cum_prob
            mayores[class_list[i]] += w_list[i] * (1 - cum_prob)

        return menores, mayores
