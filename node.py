from __future__ import division
from collections import Counter

import multiprocessing
import numpy as np

# data es un dataframe que tiene que contener una columna class. La cual el arbol intenta predecir.
# podria pensar en relajar esto y simplemente indicar cual es la variable a predecir.

class Node:
    def __init__(self, data, level = 1, max_depth = 8, min_samples_split=10):

        # Atributos particulares del nodo
        self.data = data
        self.is_leaf = False
        self.clase = ''
        self.feat_name = ""
        self.feat_value = None
        self.left = None
        self.right = None
        self.entropia = self.entropy(data)
        self.is_left = False
        self.is_right = False
        self.level = level
        self.n_rows = len(data.index)

        # Atributos generales del arbol
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        # Si es necesario particionar el nodo, llamo a split para hacerlo
        if self.check_data():
            self.split()

            # Falta corregir esto. No entiendo pq a veces el split deja el feat_name como vacio
            if self.feat_name != '':
                print 'Feature elegida: ' + self.feat_name
                menores = self.get_menores(self.feat_name, self.feat_value)
                mayores = self.get_mayores(self.feat_name, self.feat_value)                
                # menores = self.data[self.data[self.feat_name] < self.feat_value]
                # mayores = self.data[self.data[self.feat_name] >= self.feat_value]

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

            # separo el dominio en todas las posibles divisiones para obtener la division optima
            pivotes = self.get_pivotes(self.data[f], 'exact')
            # pivotes = self.get_pivotes(self.data[f], 'aprox')

            # pool = multiprocessing.Pool(processes=3)
            # ganancias = pool.map(self.pivot_gain, pivotes, [f]*len(pivotes))
            ganancias = map(self.pivot_gain, pivotes, [f]*len(pivotes))

            # Agrego el minimo valor posible de ganancia y reduzco
            ganancias.append([max_gain, None])
            pivot_max = reduce(self.maximo, ganancias)

            # Si es mejor que el maximo valor anterior agrego
            if (pivot_max[0] > max_gain):
                max_gain = pivot_max[0]
                self.feat_value = pivot_max[1]
                self.feat_name = pivot_max[2]

    # Toma un pivote y una feature, y retorna su ganancia de informacion
    def pivot_gain(self, pivote, f):
        
        # Separo las tuplas segun si su valor de esa variable es menor o mayor que el pivote
        menores = self.get_menores(f, pivote)
        mayores = self.get_mayores(f, pivote)

        # No considero caso en que todos los datos se vayan a una sola rama
        if menores.empty or mayores.empty:
            return [-float('inf'), None, None]

        # Calculo la ganancia de informacion para esta variable
        return [self.gain(menores, mayores, f), pivote, f]

    def maximo(self, a, b):
        if a[0] > b[0]:
            return a
        else:
            return b

    def get_menores(self, feature, pivote):
        return self.data[self.data[feature] < pivote]

    def get_mayores(self, feature, pivote):
        return self.data[self.data[feature] >= pivote]

    # Retorna las features a considerar en un nodo para hacer la particion
    def filterfeatures(self):
        filter_arr = []
        for f in self.data.columns:
            if not '_comp' in f and not '.l' in f and not '.r' in f and not '.std' in f and f != 'weight' and f != 'class':
                filter_arr.append(f)
        return filter_arr

    # determina se es necesario hacer un split de los datos
    def check_data(self):
        featuresfaltantes = self.filterfeatures()

        if self.data['class'].nunique() == 1 or len(featuresfaltantes) == 0:
            return False
        elif self.level >= self.max_depth:
            return False
        elif self.data.shape[0] < self.min_samples_split:
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
            for i in xrange(100):
                pivotes.append(minimo + step*i)

            return pivotes

    # Convierte el nodo en hoja. Colocando la clase mas probable como resultado
    def set_leaf(self):
        self.is_leaf = True
        # self.clase = stats.mode(self.data['class'])[0].item()
        aux = Counter(self.data['class'])
        self.clase = aux.most_common(1)[0][0]
        

    def add_left(self, left_data):
        self.left = self.__class__(left_data, self.level+1, self.max_depth, self.min_samples_split)
        self.left.is_left = True

    def add_right(self, right_data):
        self.right = self.__class__(right_data, self.level+1, self.max_depth, self.min_samples_split)
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
    # Deje la variable feature que no me sirve en la clase base, solo para ahorrarme repetir el metodo split. 
    # Eso debe poder arreglarse
    def gain(self, menores, mayores, feature):
        gain = self.entropia - (len(menores) * self.entropy(menores) + len(mayores) * self.entropy(mayores)) / self.n_rows

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