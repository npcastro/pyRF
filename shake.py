# Hace un shake dirigido especificamente a los puntos de las features que estan cerca
# de los cortes que hace un arbol normal entrenado sobre el set de entrenamiento

import pandas as pd
import tree
import pickle
import numpy as np
import random

def is_in_margin(points, value, delta):
    ret = False

    for p in points:
        if value < (p + delta) and value > (p - delta):
            ret = True

    return ret

def add_uncertainty(data, level, points_dict, delta, certainty_level = 0.1):
    """Toma un dataframe normal (cada columna es una feature y la ultima es class), 
       y le agrega incertidumbre a los ptos que esten a menos de delta de distancia
       de los puntos

    data: dataframe
    level: porcentaje del rango. Se usa como el valor maximo de incertidumbre que se le da a un punto
    points_dict: es un diccionario con una lista de valores para cada feature
    delta: representa un porcentaje del rango de las features. 
    certainty_level: porcentaje que se considera certidumbre
    """
    
    # Obtengo los rangos para cada variable. Despues la incertidumbre se pone como fraccion de estos.
    rangos = {col: (data[col].max() - data[col].min()) for col in data.columns[0:-1]}

    df = {}

    for col in data.columns[0:-1]:

        feature = data[col]
        mean = []
        std = []
        l = []
        r = []
        
        # Caso en que a esa feature no se le quiera agregar incertidumbre
        if not col in points_dict:
            for i in xrange(feature.size):

                original_value = feature.iloc[i]
                mean.append(original_value)
                std.append(0)

                # Se necesitan bored para que el metodo de la probabilidad no se caiga 
                l.append(sampled_value - (rangos[col]/2) * certainty_level)
                r.append(sampled_value + (rangos[col]/2) * certainty_level)

        else:
            split_list = points_dict[col]

            for i in xrange(feature.size):
                uncertainty = random.randrange(1, level, 1) / 100.0
                original_value = feature.iloc[i]

                # Pongo la incertidumbre correspondiente
                if is_in_margin(split_list, original_value, delta * rangos[col]):

                    sampled_value = np.random.normal(original_value, rangos[col]*uncertainty/6)
                    mean.append(sampled_value)
                    std.append(rangos[col]*uncertainty/6)
                    l.append(sampled_value - (rangos[col]/2) * uncertainty)
                    r.append(sampled_value + (rangos[col]/2) * uncertainty)

                # Pongo incertidumbre 0
                else:
                    mean.append(original_value)
                    std.append(0)

                    # Se necesitan bored para que el metodo de la probabilidad no se caiga 
                    l.append(original_value - (rangos[col]/2) * certainty_level)
                    r.append(original_value + (rangos[col]/2) * certainty_level)

            df[col + '.mean'] = mean
            df[col + '.std'] = std
            df[col + '.l'] = l
            df[col + '.r'] = r

    nuevo = pd.DataFrame(df, index = data.index)
    nuevo['weight'] = pd.Series([1.0 for i in range(len(nuevo))], index=data.index)
    nuevo['class'] = data['class']

    return nuevo

if __name__ == '__main__':

    data = pd.read_csv('sets/Macho.csv', index_col = 0)
    uncertainty_levels = range(5, 70, 5)
    points_dict = pickle.load(open('Macho splits.pkl', 'rb'))
    delta = 0.1

    for u in uncertainty_levels:
        u_data = add_uncertainty(data, u, points_dict, delta)
        u_data.to_csv('sets/Macho shaken/Macho shake ' + str(u) +'.csv', index=False)

    # Calculo el numero de tuplas con incertidumbre 0 para cada feature
    cols = [c for c in u_data.columns if '.std' in c]
    std_data = u_data[cols]
    n_zero_std = (std_data==0).sum()

    n_zero_std.to_csv('sets/Macho shaken/zero_std.csv')