import UNode
import sys
from copy import deepcopy

def eval_feature_split(feature, data, nodo):
    """Evaluates the best possible information gain for a given feature

    Parameters
    ----------
    feature: The name of the feature in the dataframe
    data: Dataframe with the features and classes
    """

    unodo = nodo
    # unodo = UNode.UNode(level = 7)
    unodo.data = data
    # unodo.entropia = unodo.entropy(data.groupby('class')['weight'].sum().to_dict())

    sys.stdout.write("\r\x1b[K" + 'Evaluando feature: ' + feature)
    sys.stdout.flush()

    # Limpio el nombre de la feature
    feature_name = feature.replace('.mean', '')

    # Ordeno el frame segun la media de la variable
    data_por_media = data.sort(feature, inplace=False)

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

    # Creo diccionarios para guardar la masa de los estrictos menores y estrictos mayores,
    # y asi no calcularla continuamente.
    # Los menores parten vacios y los mayores parten con toda la masa
    menores_estrictos_mass = {c: 0.0 for c in clases}
    mayores_estrictos_mass = data_por_media.groupby('class')['weight'].sum().to_dict()

    current_gain = -sys.maxint - 1
    current_pivot = 0

    for pivote in unodo.get_split_candidates(feature_name, split_type='otro'):

        menores_index, mayores_index = unodo.update_indexes(
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

        menores, mayores = unodo.split_tuples_by_pivot(
            w_list_afectada, mean_list_afectada, std_list_afectada,
            left_bound_list_afectada, right_bound_list_afectada, class_list_afectada,
            pivote, deepcopy(menores_estrictos_mass), deepcopy(mayores_estrictos_mass)
        )

        if not any(menores) or not any(mayores):
            continue

        elif sum(menores.values()) == 0 or sum(mayores.values()) == 0:
            continue

        # Calculo la ganancia de informacion para este pivote
        menores = unodo.fix_numeric_errors(menores)
        mayores = unodo.fix_numeric_errors(mayores)
        pivot_gain = unodo.gain(menores, mayores)

        if pivot_gain > current_gain:
            current_gain = pivot_gain
            current_pivot = pivote
    
    return current_gain, current_pivot

