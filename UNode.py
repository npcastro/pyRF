from node import *
import pandas as pd
from copy import deepcopy
import pyRF_prob

import sys
import time
import datetime

import math


class UNode(Node):
    def __init__(self, data, level=1, max_depth=8, min_samples_split=10, most_mass_threshold=0.9, min_mass_threshold=0.0127, min_weight_threshold=0.0):

        self.most_mass_threshold = most_mass_threshold
        self.min_mass_threshold = min_mass_threshold
        self.min_weight_threshold = min_weight_threshold
        mass = float(data['weight'].sum())
        Node.__init__(self, data, level, max_depth, min_samples_split, mass)
        

    def get_split_candidates(self, feature_name):
        """ Retorna todos los valores segun los que se debe intentar cortar una feature
        """
        bounds = self.data[feature_name + '.l'].unique().tolist() + self.data[feature_name + '.r'].unique().tolist()

        bounds.sort()
        print 'Numero de candidatos: ' + str(len(bounds))
        return bounds

    # def prun_homogeneus_intervals(self, bounds, feature_name):
    #     pruned = []

    #     pruned.append(bounds[0])

    #     for i in xrange(len(bounds)):

    #         if self.is_homogeneus(pruned[-1], bounds[i]):
    #             if self.is_homogeneus()
    #             continue
    #         else:
    #             pruned.append(bounds[i + 1])

    #     return pruned

    # def is_homogeneus(self, left_bound, right_bound, feature_name):
    #     filtered_data = self.data[(self.data[feature_name + '.l'] < right_bound) & (self.data[feature_name + '.r'] > left_bound)]

    #     if filtered_data.groupby('class').ngroups == 1:
    #         return True
    #     else:
    #         return False

    # def is_empty(self, left_bound, right_bound, feature_name):
    #     filtered_data = self.data[(self.data[feature_name + '.l'] > right_bound) & (self.data[feature_name + '.r'] > left_bound)]

    #     if len(filter_data.index) == 0:
    #         return True
    #     else:
    #         return False


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
            feature_name = f.rstrip('.mean')

            # Ordeno el frame segun la media de la variable
            data_por_media = self.data.sort(f, inplace=False)

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

            # Creo diccionarios para guardar la masa de los estrictos menores y estrictos mayores, y asi no calcularla continuamente.
            # Los menores parten vacios y los mayores parten con toda la masa
            menores_estrictos_mass = { c: 0.0 for c in clases}
            mayores_estrictos_mass = data_por_media.groupby('class')['weight'].sum().to_dict()
            
            # Me muevo a traves de los posibles pivotes
            for pivote in self.get_split_candidates(feature_name):
            # for pivote in mean_list:

                # Actualizo los indices
                menores_index, mayores_index = self.update_indexes(menores_index, mayores_index, pivote, left_bound_list, right_bound_list)
                # print menores_index, mayores_index

                # Actualizo la masa estrictamente menor y mayor
                for j in xrange(old_menores_index, menores_index):
                    menores_estrictos_mass[class_list[j]] += w_list[j]

                for j in xrange(old_mayores_index, mayores_index):
                    mayores_estrictos_mass[class_list[j]] -= w_list[j]

                # print old_mayores_index, mayores_index, mayores_estrictos_mass['MicroLensing']

                # Actualizo los indices anteriores
                old_menores_index, old_mayores_index = menores_index, mayores_index

                # Guardo las listas de elementos afectados por el pivote actual
                w_list_afectada = w_list[menores_index:mayores_index]
                mean_list_afectada = mean_list[menores_index:mayores_index]
                std_list_afectada = std_list[menores_index:mayores_index]
                left_bound_list_afectada = left_bound_list[menores_index:mayores_index]
                right_bound_list_afectada = right_bound_list[menores_index:mayores_index]
                class_list_afectada = class_list[menores_index:mayores_index]

                menores, mayores = self.split_tuples_by_pivot(w_list_afectada, mean_list_afectada, std_list_afectada, left_bound_list_afectada, right_bound_list_afectada, class_list_afectada, pivote, deepcopy(menores_estrictos_mass), deepcopy(mayores_estrictos_mass))

                if not any(menores) or not any(mayores):
                    continue

                elif sum(menores.values()) == 0 or sum(mayores.values()) == 0:
                    continue

                # Calculo la ganancia de informacion para esta variable
                menores, mayores = self.fix_numeric_errors(menores, mayores)
                pivot_gain = self.gain(menores, mayores)

                # En caso de error
                # if math.isnan(pivot_gain):
                #     print 'Total: ' + str(data_por_media.groupby('class')['weight'].sum().to_dict())
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
        print 'Tiempo tomado por nodo: ' + str(datetime.timedelta(0,end_time - start_time))
            # break # Para testear cuanto se demora una sola feature

    
    def fix_numeric_errors(self, menores, mayores):
        """Se rendondean a cero las masas extremadamente chicas.
        """
        for key in menores.keys():
            if abs(menores[key]) < 1e-10 and menores[key] < 0:
                menores[key] = 0

        for key in mayores.keys():
            if abs(mayores[key]) < 1e-10 and mayores[key] < 0:
                mayores[key] = 0

        return menores, mayores


    def get_relevant_columns (data, feature_name, menores_index=0, mayores_index=0):
        w_list = data['weight'].tolist()
        mean_list = data[feature_name + '.mean'].tolist()
        std_list = data[feature_name + '.std'].tolist()
        left_bound_list = data[feature_name + '.l'].tolist()
        right_bound_list = data[feature_name + '.r'].tolist()
        class_list = data['class'].tolist() 

        return w_list, mean_list, std_list, left_bound_list, right_bound_list, class_list

    # Toma los indices de los estrictamente menores y mayores, mas el nuevo pivote y los actualiza
    def update_indexes(self, menores_index, mayores_index, pivote, limites_l, limites_r):

        # Actualizo menores
        ultimo_r_menor = limites_r[menores_index]

        # Itero hasta encontrar una tupla que NO sea completamente menor que el pivote
        while( ultimo_r_menor < pivote and menores_index < len(limites_r) - 1):
            menores_index += 1
            ultimo_r_menor = limites_r[menores_index]
            
        # Actualizo mayores
        ultimo_l_mayor = limites_l[mayores_index]

        # Itero hasta encontrar una tupla que SEA completamente mayor que el pivote
        while( ultimo_l_mayor < pivote and mayores_index < len(limites_l) - 1):
            ultimo_l_mayor = limites_l[mayores_index]
            mayores_index += 1

        return menores_index, mayores_index

    def split_tuples_by_pivot(self, w_list, mean_list, std_list, left_bound_list, right_bound_list, class_list, pivote, menores, mayores):
        """
        Toma un grupo de datos lo recorre entero y retorna dos diccionarios con las sumas de masa 
        separadas por clase. Un diccionario es para los datos menores que el pivote y el otro para los mayores
        """
        clip = lambda x, l, r: l if x < l else r if x > r else x

        for i in xrange(len(class_list)):
            cum_prob = pyRF_prob.cdf(pivote, mean_list[i], std_list[i], left_bound_list[i], right_bound_list[i])

            # if cum_prob > 1 or cum_prob < 0:
            #     print cum_prob

            cum_prob = clip(cum_prob, 0, 1)

            menores[class_list[i]] += w_list[i] * cum_prob
            mayores[class_list[i]] += w_list[i] * (1 - cum_prob)

        return menores, mayores    


    def gain(self, menores, mayores):
        """ Retorna la ganancia de dividir los datos en menores y mayores
            Menores y mayores son diccionarios donde la llave es el nombre 
            de la clase y los valores son la suma de masa para ella.
        """
        gain = self.entropia - ( sum(menores.values()) * self.entropy(menores) + sum(mayores.values()) * self.entropy(mayores) ) / self.mass

        return gain

    def entropy(self, data):
        """ Retorna la entropia de un grupo de datos.
        data: diccionario donde las llaves son nombres de clases y los valores sumas (o conteos de valores)
        """

        total = sum(data.values())
        entropia = 0
        
        for clase in data.keys():
            if data[clase] != 0:
                entropia -= (data[clase] / total) * np.log2(data[clase] / total)

        return entropia

    def add_left(self, left_data):
        self.left = self.__class__(left_data, self.level+1, self.max_depth, self.min_samples_split, self.most_mass_threshold)
        self.left.is_left = True

    def add_right(self, right_data):
        self.right = self.__class__(right_data, self.level+1, self.max_depth, self.min_samples_split, self.most_mass_threshold)
        self.right.is_right = True
    
    def predict(self, tupla, prediction={}, w=1):
        # Si es que es el nodo raiz
        if len(prediction.keys()) == 0:
            prediction = {c: 0.0 for c in self.data['class'].unique() }

        if self.is_leaf:
            aux = deepcopy(prediction)
            aux[self.clase] += w
            return aux

        # Puede que falte chequear casos bordes, al igual que lo hago en get_menores y get_mayores
        else:
            feature_name = self.feat_name.rstrip('.mean')
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

    def get_menores(self, feature_name, pivote):
        menores = []

        # limpio el nombre de la feature
        feature_name = feature_name.rstrip('.mean')

        menores = self.data[self.data[feature_name + '.l'] < pivote]
        # self.data.apply(func=self.get_weight, axis=1, args=[menores, pivote, feature_name, "menor"])

        menores = menores.apply(func=self.get_weight, axis=1, args=[pivote, feature_name, "menor"])
        # menores = menores[menores["weight"] != 0]
        menores = menores[menores["weight"] > self.min_weight_threshold]
                 
        return pd.DataFrame(menores, index = menores.index)


    def get_mayores(self, feature_name, pivote):
        mayores = []

        # limpio el nombre de la feature
        feature_name = feature_name.rstrip('.mean')

        mayores = self.data[self.data[feature_name + '.r'] >= pivote]
        # self.data.apply(func=self.get_weight, axis=1, args=[mayores, pivote, feature_name, "mayor"])
  
        mayores = mayores.apply(func=self.get_weight, axis=1, args=[pivote, feature_name, "mayor"])
        # mayores = mayores[mayores["weight"] != 0]  
        mayores = mayores[mayores["weight"] > self.min_weight_threshold]  
  
        return pd.DataFrame(mayores, index = mayores.index)

    # Convierte el nodo en hoja. Colocando la clase mas probable como resultado
    def set_leaf(self):
        self.is_leaf = True
        self.clase = self.data.groupby('class')['weight'].sum().idxmax()


    def get_weight(self, tupla, pivote, feature_name, how):
        """ Determina la distribucion de probabilidad gaussiana acumulada entre dos bordes.
        
        pivote: valor de corte
        how: determina si la probabilidad se calcula desde l hasta pivote o desde pivote hasta r
        """

        left_bound = tupla[feature_name+'.l']
        right_bound = tupla[feature_name+'.r']

        if left_bound >= pivote and how == 'mayor' or right_bound <= pivote and how == 'menor':
            return tupla

        else:
            w = tupla['weight']
            mean = tupla[feature_name+'.mean']
            std = tupla[feature_name+'.std']
   
   
            feature_mass = pyRF_prob.cdf(pivote, mean, std, left_bound, right_bound)

            if how == 'menor':
                if (feature_mass >= self.min_mass_threshold):
                    tupla['weight'] = min(w * feature_mass, 1)
                else:
                    tupla['weight'] = 0
                # tupla[feature_name+'.r'] = min(pivote, tupla[feature_name + '.r'])
                tupla[feature_name+'.r'] = pivote
                return tupla

            elif how == 'mayor':
                feature_mass = 1 - feature_mass
                if (feature_mass>= self.min_mass_threshold):
                    tupla['weight'] = min(w * feature_mass, 1)
                else:
                    tupla['weight'] = 0
                # tupla[feature_name+'.l'] = max(pivote, tupla[feature_name + '.l'])
                tupla[feature_name+'.l'] = pivote
                return tupla
 
    def check_leaf_condition(self):
        """ Determina se es necesario hacer un split de los datos
        """
        featuresfaltantes = self.filterfeatures()

        if self.data['class'].nunique() == 1 or len(featuresfaltantes) == 0:
            return False
        elif self.level >= self.max_depth:
            return False
        elif self.n_rows < self.min_samples_split:  #Creo que esta condicion esta de mas. La de abajo ya lo abarca y mejor
            return False
        elif self.mass < self.min_samples_split:    
            return False
        elif self.check_most_mass():
            return False
        else:
            return True

    def check_most_mass(self):
        """ Determina si el nodo tiene masa asociada a un clase lo suficientemente representativa segun un threshold
        """

        mass_sum = self.data.groupby('class')['weight'].sum().to_dict()
        
        for clase in mass_sum.keys():
            if mass_sum[clase]/self.mass  >= self.most_mass_threshold:
                return True

        return False