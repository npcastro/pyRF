import tree
from node import *
from UNode import *
import pandas as pd

import cProfile as cp
import pstats
import timeit


# Leo los datos
path = "sets/macho 20.csv"
data = pd.read_csv(path)

# Agrego el peso de las tuplas
data['weight'] = data['weight'].astype(float)

# Para testeo rapido
data = data[0:500]

##################### Node.py ##########################
nodo = Node(data, level = 10) # Con 10 evito que el nodo crezca

f = nodo.filterfeatures()[0]	
p = nodo.get_pivotes(data[f], 'exact')  
# menores = nodo.get_menores(f, p[100])
# mayores = nodo.get_mayores(f, p[100])


##################### UNode.py ##########################
unodo = UNode(data, level = 10)

# cp.run('unodo.get_menores(f,p[100])', 'restats')
# p = pstats.Stats('restats')
# p.sort_stats('time').print_stats(10)







#!/usr/bin/python
# import pyRF_prob
# import scipy.stats

# mean = 0
# std = 1

# def nueva():
# 	return pyRF_prob.cdf(0.333,mean,std,-1,1)

# def antigua():
# 	total_mass = scipy.stats.norm(mean, std).cdf(1) - scipy.stats.norm(mean, std).cdf(-1)
# 	pivot_mass = scipy.stats.norm(mean, std).cdf(0.333) - scipy.stats.norm(mean, std).cdf(-1)
# 	return (pivot_mass / total_mass)

# def test():
# 	total_mass = scipy.stats.norm(mean, std).cdf(1) - scipy.stats.norm(mean, std).cdf(-1)
# 	pivot_mass = scipy.stats.norm(mean, std).cdf(1) - scipy.stats.norm(mean, std).cdf(0.333)
# 	return (pivot_mass / total_mass)

#scipy.stats.norm(mean, std).cdf(pivote)

# import scipy.stats