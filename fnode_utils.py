# coding=utf-8

import sys
from copy import deepcopy

import numpy as np

import FNode
import pyRF_prob


def check_unique_presence(values):
    """Check if there's a class with presence and all else with zero"""
    aux = set(values)

    if 0 in aux and len(aux) == 2:
        return True
    else:
        return False


def eval_feature(name_data_tuple, entropia, mass):
    """Evaluates the best possible information gain for a given feature

    Parameters
    ----------
    name_data_tuple: tuple containing the name of the feature, and the data corresponding to it
    entropia: the current entropy of the node that it's being evaluated
    mass: the total mass of the tuples of the node that it's being evaluated
    """

    feature_name, data = name_data_tuple

    # print 'Evaluando feature: ' + feature_name

    # Ordeno el frame segun la media de la variable
    data_por_media = data.sort(feature_name + '.mean', inplace=False)

    # Transformo la informacion relevante de esta feature a listas
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

    current_gain = -sys.maxint - 1
    # current_gain = 0
    current_pivot = 0

    for pivote in get_split_candidates(data_por_media, feature_name, split_type='otro'):

        menores_index, mayores_index = update_indexes(
            menores_index, mayores_index,
            pivote, left_bound_list, right_bound_list
        )

        # Actualizo la masa estrictamente menor y mayor
        for j in xrange(old_menores_index, menores_index):
            menores_estrictos_mass[class_list[j]] += w_list[j]

        for j in xrange(old_mayores_index, mayores_index):
            mayores_estrictos_mass[class_list[j]] -= w_list[j]

        # Guardo los indices actuales
        old_menores_index, old_mayores_index = menores_index, mayores_index

        # Guardo las listas de elementos afectados por el pivote actual
        w_list_afectada = w_list[menores_index:mayores_index]
        mean_list_afectada = mean_list[menores_index:mayores_index]
        std_list_afectada = std_list[menores_index:mayores_index]
        left_bound_list_afectada = left_bound_list[menores_index:mayores_index]
        right_bound_list_afectada = right_bound_list[menores_index:mayores_index]
        class_list_afectada = class_list[menores_index:mayores_index]

        menores, mayores = split_tuples_by_pivot(
            w_list_afectada, mean_list_afectada, std_list_afectada,
            left_bound_list_afectada, right_bound_list_afectada, class_list_afectada,
            pivote, deepcopy(menores_estrictos_mass), deepcopy(mayores_estrictos_mass)
        )

        if not any(menores) or not any(mayores):
            continue

        elif sum(menores.values()) == 0 or sum(mayores.values()) == 0:
            continue

        # Calculo la ganancia de informacion para este pivote
        menores = fix_numeric_errors(menores)
        mayores = fix_numeric_errors(mayores)
        pivot_gain = gain(menores, mayores, entropia, mass)

        if pivot_gain > current_gain:
            current_gain = pivot_gain
            current_pivot = pivote

    return current_gain, current_pivot


def fix_numeric_errors(num_dict):
    """Masses that are extremely small are rounded to zero."""

    for key in num_dict.keys():
        if abs(num_dict[key]) < 1e-10 and num_dict[key] < 0:
            num_dict[key] = 0

    return num_dict


def entropy(data):
    """Calculates the entropy of a group of data
    data: dicctionary where the keys are class names, and the values are counts or sums of mass
    """

    total = float(sum(data.values()))
    entropia = 0

    for clase in data.keys():
        if data[clase] != 0:
            entropia -= (data[clase] / total) * np.log2(data[clase] / total)

    return entropia


def gain(menores, mayores, entropia, masa):
    """Retorna la ganancia de dividir los datos en menores y mayores

    Menores y mayores son diccionarios donde la llave es el nombre
    de la clase y los valores son la suma de masa para ella.
    """
    gain = (entropia - (sum(menores.values()) * entropy(menores) +
            sum(mayores.values()) * entropy(mayores)) / masa)

    return gain


# Parece que estoy guardando la clase actual por las puras
def get_class_changes(left_values, right_values, clases):
    presence = {c: 0 for c in set(clases)}
    bounds = []

    left_index = 1
    right_index = 0

    # I add the values for the first point (neccesarily a left bound)
    clase_actual = clases[0]
    presence[clase_actual] = 1

    while right_index < len(right_values):

        if left_index < len(left_values) and left_values[left_index] <= right_values[right_index]:
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
        elif check_unique_presence(presence.values()):
            continue

        else:
            bounds.append(value)

        if right:
            presence[clase_actual] -= 1

    return bounds


def get_split_candidates(data, feature_name, split_type='simple'):
    """Returns a list of all the points of a feature that must be tested as a split point
    """
    if split_type == 'simple':
        bounds = (data[feature_name + '.l'].tolist() +
                  data[feature_name + '.r'].tolist())

        # print 'Splits metodo simple: ' + str(len(np.unique(bounds)))
        return np.unique(bounds)

    else:
        bounds = get_class_changes(data[feature_name + '.l'].tolist(),
                                   data[feature_name + '.r'].tolist(),
                                   data['class'].tolist())
        bounds = np.unique(bounds)
        print 'Splits metodo nuevo: ' + str(len(bounds))
        return bounds


def split_tuples_by_pivot(w_list, mean_list, std_list, left_bound_list, right_bound_list,
                          class_list, pivote, menores, mayores):
        """divides a group of data according to a pivot

        It operates along all the data. And then returns two dictionaries with the total sum
        of the mass separated by class.

        Returns:
            menores: Dictionary for the data thats inferior than the pivot
            mayores: Dictionary for the data thats superior to the pivot
        """
        clip = lambda x, l, r: l if x < l else r if x > r else x

        for i in xrange(len(class_list)):
            cum_prob = pyRF_prob.cdf(pivote, mean_list[i], std_list[i], left_bound_list[i],
                                     right_bound_list[i])

            cum_prob = clip(cum_prob, 0, 1)

            menores[class_list[i]] += w_list[i] * cum_prob
            mayores[class_list[i]] += w_list[i] * (1 - cum_prob)

        return menores, mayores


def update_indexes(menores_index, mayores_index, pivote, limites_l, limites_r):
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
