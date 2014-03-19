from __future__ import division
from collections import Counter

import numpy as np

# data es un dataframe que tiene que contener una columna class. La cual el arbol intenta predecir.
# podria pensar en relajar esto y simplemente indicar cual es la variable a predecir.

class Node:
    def __init__(self, data, criterium, level = 1):

        self.data = data
        self.is_leaf = False
        self.clase = ''
        self.feat_name = ""
        self.feat_value = None
        self.left = None
        self.right = None
        self.criterium = criterium
        self.entropia = self.entropy(data)
        self.level = level
        self.is_left = False
        self.is_right = False

        # Si es necesario particionar el nodo, llamo a split para hacerlo
        if self.check_data():
            self.split()

            # Falta corregir esto. No entiendo pq a veces el split deja el feat_name como vacio
            if self.feat_name != '':
                print 'Feature elegida: ' + self.feat_name
                menores = self.data[self.data[self.feat_name] < self.feat_value]
                mayores = self.data[self.data[self.feat_name] >= self.feat_value]

<<<<<<< HEAD
                #menores = menores.drop(self.feat_name, 1)
                #mayores = mayores.drop(self.feat_name, 1)

                #if self.criterium == 'confianza':
                    #menores = menores.drop(self.feat_name + '_comp', 1)
                    #mayores = mayores.drop(self.feat_name + '_comp', 1)
=======
                # menores = menores.drop(self.feat_name, 1)
                # mayores = mayores.drop(self.feat_name, 1)

                # if self.criterium == 'confianza':
                #     menores = menores.drop(self.feat_name + '_comp', 1)
                #     mayores = mayores.drop(self.feat_name + '_comp', 1)
>>>>>>> ef2ead75d86ed458cc05352d301dc95d1ccb2bf2

                if not menores.empty:
                    self.add_left(menores)
                if not mayores.empty:
                    self.add_right(mayores)

            else:
                self.set_leaf()

        # De lo contrario llamo a set_leaf para transformarlo en hoja
        else:
            self.set_leaf()

    # Busca el mejor corte posible para el nodo
    def split(self):
        # Inicializo la ganancia de info en el peor nivel posible
        max_gain = -float('inf')

        # Para cada feature (no considero la clase ni la completitud)
        filterfeatures = self.filterfeatures()

        print filterfeatures

        for f in filterfeatures:
            print 'Evaluando feature: ' + f

            # separo el dominio en todas las posibles divisiones para obtener la optima division
            pivotes = self.get_pivotes(self.data[f], 'exact')
            # pivotes = self.get_pivotes(self.data[f], 'aprox')

            for pivote in pivotes:

                # Separo las tuplas segun si su valor de esa variable es menor o mayor que el pivote
                menores = self.data[self.data[f] < pivote]
                mayores = self.data[self.data[f] >= pivote]

                # No considero caso en que todos los datos se vayan a una sola rama
                if menores.empty or mayores.empty:
                    continue

                # Calculo la ganancia de informacion para esta variable
                if self.criterium == 'gain':
                    gain = self.gain(menores, mayores)
                elif self.criterium == 'confianza':
                    gain = self.confianza(menores, mayores, f)

                # Comparo con la ganancia anterior, si es mejor guardo el gain, la feature correspondiente y el pivote
                if (gain > max_gain):
                    max_gain = gain
                    self.feat_name = f
                    self.feat_value = pivote

    def filterfeatures(self):
        # Para cada feature (no considero la clase ni la completitud)
        filterfeatures = []
        for feature in self.data.columns:
            if self.criterium == 'gain' and not '_comp' in feature and feature is not 'class':
                filterfeatures.append(feature)
            elif self.criterium == 'confianza' and not '_comp' in feature and feature != 'class':
                filterfeatures.append(feature)
        return filterfeatures


    # determina se es necesario hacer un split de los datos
    def check_data(self):
        featuresfaltantes = self.filterfeatures()

        if self.data['class'].nunique() == 1 or len(featuresfaltantes) == 0:
            return False
        elif self.level >= 8:
            return False
        else:
            return True

    # retorna una lista con los todos los threshold a evaluar para buscar la mejor separacion
    def get_pivotes(self, feature, calidad = 'exact'):

        if calidad == 'exact':
            return feature[1:].unique()
        elif calidad == 'aprox':
            minimo = feature.min()
            maximo = feature.max()
            step = maximo - minimo / 100
            pivotes = []
            for i in range(100):
                pivotes.append(minimo + step*i)

            return pivotes

    # Convierte el nodo en hoja. Colocando la clase mas probable como resultado
    def set_leaf(self):
        self.is_leaf = True
        # self.clase = stats.mode(self.data['class'])[0].item()
        aux = Counter(self.data['class'])
        self.clase = aux.most_common(1)[0][0]
        

    def add_left(self, left_data):
        self.left = Node(left_data, self.criterium, self.level+1)
        self.left.is_left = True

    def add_right(self, right_data):
        self.right = Node(right_data, self.criterium, self.level+1)
        self.right.is_right = True

    def predict(self, tupla, confianza=1):
        if self.is_leaf:
            return self.clase, confianza
        else:
            if tupla[self.feat_name] < self.feat_value:
                if self.criterium == 'confianza':
                    # Propago la incertidumbre del dato que estoy prediciendo
                    # return self.left.predict(tupla, confianza * tupla[self.feat_name + '_comp'])
                    return self.left.predict(tupla, (confianza + tupla[self.feat_name + '_comp'])/2)
                else:
                    return self.left.predict(tupla)
            else:
                if self.criterium == 'confianza':
                    # return self.right.predict(tupla, confianza * tupla[self.feat_name + '_comp'])
                    return self.right.predict(tupla, (confianza + tupla[self.feat_name + '_comp'])/2)
                else:
                    return self.right.predict(tupla)

    def show(self, linea=""):
        if self.is_leaf:
            print linea + '|---- ' + str(self.clase)

        elif self.is_left:
            self.right.show(linea + '|     ')
            print linea + '|- '+ self.feat_name + ' ' + '(' + ("%.2f" % self.feat_value) + ')'
            self.left.show(linea + '      ')

        elif self.is_right:
            self.right.show(linea + '      ')
            print linea + '|- ' + self.feat_name + ' ' + '(' + ("%.2f" % self.feat_value) + ')'
            self.left.show(linea + '|     ')

        # Es el nodo raiz
        else:
            self.right.show(linea + '      ')
            print linea + '|- '+ self.feat_name + ' ' + '(' + ("%.2f" % self.feat_value) + ')'
            self.left.show(linea + '      ')  

    
    # Retorna la ganancia de dividir los datos en menores y mayores.
    def gain(self, menores, mayores):

        total = len(self.data.index)

        gain = self.entropia - (len(menores) * self.entropy(menores) + len(mayores) * self.entropy(mayores)) / total

        return gain

    # Retorna la entropia de un grupo de datos
    def entropy(self, data):
        clases = data['class'].unique()
        total = len(data.index)

        entropia = 0

        for c in clases:
            p_c = len(data[data['class'] == c].index) / total
            entropia -= p_c * np.log2(p_c)

        return entropia


    def confianza(self, menores, mayores, feature):
        total = sum(menores[feature + '_comp']) + sum(mayores[feature + '_comp'])

        confianza = self.entropia - (
            sum(menores[feature + '_comp']) * self.trust(menores, feature) + sum(
                mayores[feature + '_comp']) * self.trust(
                mayores, feature)) / total

        return confianza

    # Retorna la entropia, calculada con confianza, de un grupo de datos en una variable.
    def trust(self, data, feature):

        clases = data['class'].unique()
        total = sum(data[feature + '_comp'])

        trust = 0

        for c in clases:
            p_c = sum(data[data['class'] == c][feature + '_comp']) / total
            trust -= p_c * np.log2(p_c)

        return trust
