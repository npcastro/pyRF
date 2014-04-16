import tree
from node import *
import pandas as pd

path = "sets/macho 20.csv"

data = pd.read_csv(path)
data['weight'] = data['weight'].astype(float)

nodo = Node(data, level = 10) # Con 10 evito que el nodo crezca

f = nodo.filterfeatures()[0]	#Una feature

p = nodo.get_pivotes(data[f], 'exact')  #pivotes

menores = nodo.get_menores(f, p[200])
mayores = nodo.get_mayores(f, p[200])