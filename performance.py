import tree
from node import *
import pandas as pd

# Leo los datos
path = "sets/macho 20.csv"
data = pd.read_csv(path)

# Agrego el peso de las tuplas
data['weight'] = data['weight'].astype(float)

# Transformo la clase a numeros
# clases = pd.Categorical(data['class'])
# nombres_clases = clases.levels[0]		# Los nombres de las clases estan en un arreglo con correspondencia a los numeros que les asignan
# data['class'] = pd.Series(clases.labels)


nodo = Node(data, level = 10) # Con 10 evito que el nodo crezca

f = nodo.filterfeatures()[0]	#Una feature

p = nodo.get_pivotes(data[f], 'exact')  #pivotes

menores = nodo.get_menores(f, p[200])
mayores = nodo.get_mayores(f, p[200])