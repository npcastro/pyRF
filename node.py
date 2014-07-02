from __future__ import division
from collections import Counter


import numpy as np


# data es un dataframe que tiene que contener una columna class. La cual el arbol intenta predecir.
# podria pensar en relajar esto y simplemente indicar cual es la variable a predecir.
class Node:
    def __init__(self, data, level = 1, max_depth = 8, min_samples_split=10, mass=None):

        # Atributos particulares del nodo

        self.data = data
        self.is_leaf = False
        self.clase = ''
        self.feat_name = ""
        self.feat_value = None
        self.left = None
        self.right = None
        self.entropia = self.entropy(data.groupby('class')['weight'].sum().to_dict())
        self.is_left = False
        self.is_right = False
        self.level = level
        self.n_rows = len(data.index)
        if mass is None:
            self.mass = float(len(data.index))
        else:
            self.mass = mass

        # Atributos generales del arbol
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        # Si es necesario particionar el nodo, llamo a split para hacerlo
        if self.check_leaf_condition():
            self.split()

            # Ojo con esto. No entiendo pq a veces el split deja el feat_name como vacio
            if self.feat_name != '':
                print 'Feature elegida: ' + self.feat_name
                print 'Pivote elegido: ' + str(self.feat_value)

                menores = self.get_menores(self.feat_name, self.feat_value)
                mayores = self.get_mayores(self.feat_name, self.feat_value)

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
        max_gain = 0

        # Para cada feature (no considero la clase ni la completitud)
        filterfeatures = self.filterfeatures()

        print filterfeatures

        pivot_gain = self.pivot_gain
        get_pivotes = self.get_pivotes

        for f in filterfeatures:
            print 'Evaluando feature: ' + f

            # Ordeno el frame segun la feature indicada
            self.data.sort(f, inplace=True)

            for i in xrange(1,self.n_rows):

                menores = self.data[0:i]
                mayores = self.data[i:]
                pivote = self.data.at[i,f]

                # if menores.empty or mayores.empty:
                #     continue

                # Calculo la ganancia de informacion para esta variable
                pivot_gain = self.gain(menores, mayores, f)

                if pivot_gain > max_gain:
                    max_gain = pivot_gain
                    self.feat_value = pivote
                    self.feat_name = f

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
    def check_leaf_condition(self):
        featuresfaltantes = self.filterfeatures()

        if self.data['class'].nunique() == 1 or len(featuresfaltantes) == 0:
            return False
        elif self.level >= self.max_depth:
            return False
        elif self.n_rows < self.min_samples_split:
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
        # self.clase = Counter(self.data['class']).most_common(1)[0][0]
        self.clase = self.data['class'].value_counts().idxmax()
        

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
        gain = self.entropia - (len(menores.index) * self.entropy(menores) + len(mayores.index) * self.entropy(mayores)) / self.n_rows

        return gain

    # # Retorna la entropia de un grupo de datos
    # def entropy(self, data):
        
    #     total = len(data.index)
    #     entropia = 0
    #     log = np.log2

    #     # clases = data['class'].unique()
        
    #     # for c in clases:
    #     #     p_c = len(data[data['class'] == c].index) / total
    #     #     entropia -= p_c * log(p_c)

    #     g = data.groupby('class')
    #     for count in g.size():
    #         entropia -= (count / total) * log(count / total)

    #     # Enfoque para UNode tbn
    #     # pesos = data.groupby('class')['weight']
    #     # for suma in pesos.sum():
    #     #     entropia -= (suma / total) * log(suma / total)

    #     return entropia

    def entropy(self, data):
        """
        Retorna la entropia de un grupo de datos.
        data: diccionario donde las llaves son nombres de clases y los valores sumas (o conteos) de valores.
        """

        total = sum(data.values())
        entropia = 0
        
        for clase in data.keys():
            entropia -= (data[clase] / total) * np.log(data[clase] / total)

        return entropia
