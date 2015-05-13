# coding=utf-8

import sys
from copy import deepcopy
from functools import partial

import numpy as np

import PNode
import pyRF_prob


def check_unique_presence(values):
    """Check if there's a class with presence and all else with zero"""
    aux = set(values)

    if 0 in aux and len(aux) == 2:
        return True
    else:
        return False


def eval_feature_split(feature, data, nodo):
    """Evaluates the best possible information gain for a given feature

    Parameters
    ----------
    feature: The name of the feature in the dataframe
    data: Dataframe with the features and classes
    """

    unodo = nodo

    print 'Evaluando feature: ' + feature

    # Limpio el nombre de la feature
    feature_name = feature.replace('.mean', '')

    # Ordeno el frame segun la media de la variable
    data_por_media = data.sort(feature, inplace=False)

    # Transformo la informacion relevante de esta feature a listas
    w_list = data_por_media['weight'].tolist()
    mean_list = data_por_media[feature_name + '.mean'].tolist()
    std_list = data_por_media[feature_name + '.std'].tolist()
    left_bound_list = data_por_media[feature_name + '.l'].tolist()
    right_bound_list = data_por_media[feature_name + '.r'].tolist()
    class_list = data_por_media['class'].tolist()

    current_gain = -sys.maxint - 1
    current_pivot = 0

    pivotes = get_split_candidates(data_por_media, feature_name, split_type='otro')
    partial_eval = partial(evaluate_split, entropia=unodo.entropia, mass=unodo.mass, w_list=w_list,
                           mean_list=mean_list, std_list=std_list, left_bound_list=left_bound_list,
                           right_bound_list=right_bound_list, class_list=class_list)

    # pool = Pool(processes=self.n_jobs)
    # gains_pivots_tuples = pool.map(partial_eval, candidate_features, 1)
    # pool.close()
    # pool.join()

    gains_pivots_tuples = map(partial_eval, pivotes)
    gains, pivots = map(list, zip(*gains_pivots_tuples))

    max_gain = 0
    pivot = 0

    for i, gain in enumerate(gains):
        if gain > max_gain:
            max_gain = gain
            pivot = pivotes[i]

    return max_gain, pivot

def evaluate_split(pivote, entropia, mass, w_list, mean_list, std_list, left_bound_list,
                   right_bound_list, class_list):

    menores, mayores = split_tuples_by_pivot(w_list, mean_list, std_list, left_bound_list,
                                                 right_bound_list, class_list, pivote)

    if not any(menores) or not any(mayores):
        return (0,0)

    elif sum(menores.values()) == 0 or sum(mayores.values()) == 0:
        return (0,0)

    menores = fix_numeric_errors(menores)
    mayores = fix_numeric_errors(mayores)
    pivot_gain = gain(menores, mayores, entropia, mass)

    current_gain = 0
    current_pivot = 0

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

        print 'Splits metodo simple: ' + str(len(np.unique(bounds)))
        return np.unique(bounds)

    else:
        bounds = get_class_changes(data[feature_name + '.l'].tolist(),
                                   data[feature_name + '.r'].tolist(),
                                   data['class'].tolist())
        bounds = np.unique(bounds)
        print 'Splits metodo nuevo: ' + str(len(bounds))
        return bounds

def split_tuples_by_pivot(w_list, mean_list, std_list, left_bound_list, right_bound_list,
                          class_list, pivote):
        """divides a group of data according to a pivot

        It operates along all the data. And then returns two dictionaries with the total sum
        of the mass separated by class.

        Returns:
            menores: Dictionary for the data thats inferior than the pivot
            mayores: Dictionary for the data thats superior to the pivot
        """
        clip = lambda x, l, r: l if x < l else r if x > r else x

        clases = set(class_list)
        menores = {c: 0.0 for c in clases}
        mayores = {c: 0.0 for c in clases}

        for i in xrange(len(class_list)):
            cum_prob = pyRF_prob.cdf(pivote, mean_list[i], std_list[i], left_bound_list[i],
                                     right_bound_list[i])

            cum_prob = clip(cum_prob, 0, 1)

            menores[class_list[i]] += w_list[i] * cum_prob
            mayores[class_list[i]] += w_list[i] * (1 - cum_prob)

        return menores, mayores
