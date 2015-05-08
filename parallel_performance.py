# coding=utf-8
# En este script voy a desarrollar y medir la paralelizacion del árbol de decision

from UNode import *
import pandas as pd
import cProfile

path = 'sets/Macho random II/Macho random 20.csv'
data = pd.read_csv(path)

unodo = UNode(level = 10)

pivote = 1.69
feature_name = 'Psi_eta'

data_por_media = data.sort(feature_name + '.mean', inplace=False)

w_list = data_por_media['weight'].tolist()
mean_list = data_por_media[feature_name + '.mean'].tolist()
std_list = data_por_media[feature_name + '.std'].tolist()
left_bound_list = data_por_media[feature_name + '.l'].tolist()
right_bound_list = data_por_media[feature_name + '.r'].tolist()
class_list = data_por_media['class'].tolist()

a, b = unodo.update_indexes(0, 0, pivote, left_bound_list, right_bound_list)

tuplas_afectadas_por_pivote = data_por_media[a:b]
w_list_afectada = w_list[a:b]
mean_list_afectada = mean_list[a:b]
std_list_afectada = std_list[a:b]
left_bound_list_afectada = left_bound_list[a:b]
right_bound_list_afectada = right_bound_list[a:b]
class_list_afectada = class_list[a:b]

menores_estrictos_mass = { c: 0.0 for c in set(class_list)}
mayores_estrictos_mass = data_por_media.groupby('class')['weight'].sum().to_dict()

for i in xrange(a):
    menores_estrictos_mass[class_list[i]] += w_list[i]

for i in xrange(b):
    mayores_estrictos_mass[class_list[i]] -= w_list[i]

menores, mayores = unodo.split_tuples_by_pivot(
    w_list_afectada, mean_list_afectada, std_list_afectada,
    left_bound_list_afectada, right_bound_list_afectada, class_list_afectada,
    pivote, deepcopy(menores_estrictos_mass), deepcopy(mayores_estrictos_mass)
)

cProfile.run('unodo.split_tuples_by_pivot(w_list_afectada, mean_list_afectada, std_list_afectada,\
    left_bound_list_afectada, right_bound_list_afectada, class_list_afectada,\
    pivote, deepcopy(menores_estrictos_mass), deepcopy(mayores_estrictos_mass))')