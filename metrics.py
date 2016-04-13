# coding=utf-8

import pandas as pd
import numpy as np


def result_to_frame(result):
    """Recibe una tabla de resultados (id, original, predicted, trust) y la mapea a un dataframe
    o matriz
    """
    lc_ids = result.index.tolist()
    unique = np.unique(np.concatenate((result['original'].values, result['predicted'].values), axis=1))
    df = pd.DataFrame(index=lc_ids, columns=unique).fillna(0.0)

    for index, row in result.iterrows():
        df.set_value(index, row['predicted'], row['trust'])

    return df

def matrix_to_result(matrix, y):
    row_sum = matrix.sum(axis=1)          # El total de las votaciones para la curva
    row_max_class = matrix.idxmax(axis=1) # La clase mas probable
    row_max_count = matrix.max(axis=1)    # La votación de la clase mas probable

    aux_dict = {}
    aux_dict['original'] = y
    aux_dict['predicted'] = row_max_class
    aux_dict['trust'] = row_max_count / row_sum

    agg_preds = pd.DataFrame(aux_dict, index=matrix.index)

    return agg_preds

def aggregate_predictions(results):
    """
    Toma una lista de resultados. Cada resultado es un dataframe con la clase original y su
    predicción para un grupo de curvas. Para cada curva junta las predicciones y retorna
    un solo dataframe con la probabilidad agregada de pertencer a la clase mas probable.
    """

    # Lista de diccionarios. Donde cada entrada corresponde a una curva y es un diccionario donde
    # las llaves son clases y contienen el conteo para cada una de las clases

    aggregate_matrix = reduce(lambda x, y: x+y, map(result_to_frame, results))

    row_sum = aggregate_matrix.sum(axis=1)          # El total de las votaciones para la curva
    row_max_class = aggregate_matrix.idxmax(axis=1) # La clase mas probable
    row_max_count = aggregate_matrix.max(axis=1)    # La votación de la clase mas probable

    aux_dict = {'original': results[0]['original']}
    aux_dict['predicted'] = row_max_class
    aux_dict['trust'] = row_max_count / row_sum

    agg_preds = pd.DataFrame(aux_dict, index=results[0].index)

    return agg_preds

def predict_table(clf, test_X, test_y):
    """Toma un random Forest entrenado y un grupo de tuplas de testing y genera un dataframe con 
    la forma df['original', 'predicted', 'trust']
    """
    
    prediction = clf.predict(test_X).tolist()
    probs = clf.predict_proba(test_X).tolist()

    tabla = []

    for index, p in enumerate(probs):
        clase = test_y.iloc[index]
        predicted = prediction[index]
        confianza = max(p)
        # confianza = 1.0
        tabla.append([clase, predicted, confianza])

    return pd.DataFrame(tabla, index=test_y.index, columns=['original', 'predicted', 'trust'])

def confusion_matrix(table):
    """Generates a confusion matrix from the prediction table"""

    unique = np.unique(np.concatenate((table['original'].values,
                       table['predicted'].values), axis=1))

    matrix = np.zeros((len(unique), len(unique)))
    matrix = pd.DataFrame(matrix)
    matrix.columns = unique
    matrix.index = unique

    for index, row in table.iterrows():
        matrix.loc[row[0]][row[1]] += row[2]

    return matrix

def hard_matrix(table):
    """Generates a hard_confusion matrix for probabilistic classifiers"""

    unique = np.unique(np.concatenate((table['original'].values, table['predicted'].values)))

    matrix = np.zeros((len(unique), len(unique)))
    matrix = pd.DataFrame(matrix)
    matrix.columns = unique
    matrix.index = unique

    for index, row in table.iterrows():
        matrix[row[0]][row[1]] += 1

    return matrix

def precision(matrix, clase):
    """Shows the accuracy of a given class, based on a confusion matrix"""

    if clase in matrix.columns:
        correctos = matrix[clase].loc[clase]
        total = matrix[clase].sum()

        return correctos / total

    # A negative number is returned to show that there are no predictions
    # of the given class on the confusion matrix
    else:
        return -1

def recall(matrix, clase):
    """Shows the recall of a given class, based on a confusion matrix"""

    if clase in matrix.columns:
        reconocidos = matrix[clase].loc[clase]
        total = matrix.loc[clase].sum()

        return reconocidos / total

    # A negative number is returned to show that there are no predictions
    # of the given class on the confusion matrix
    else:
        return -1

def f_score(matrix, clase):
    """Shows the f_score of a given class, based on a confusion matrix"""

    acc = precision(matrix, clase)
    rec = recall(matrix, clase)

    # Neccesary check, in order to avoid divisions by zero
    if acc == 0 or rec == 0:
        return 0

    # Check that both are valid
    elif acc == -1 or rec == -1:
        return -1

    # Retorno f_score
    else:
        return 2 * acc * rec / (acc + rec)

def weighted_f_score(matrix):
    clases = matrix.columns.tolist()

    counts = {c: matrix.loc[c].sum() for c in clases}
    f_scores = {c: f_score(matrix, c) for c in clases}

    total = float(sum(counts.values()))

    ret = 0
    for c in clases:
        ret += counts[c] / total * f_scores[c]

    return ret
