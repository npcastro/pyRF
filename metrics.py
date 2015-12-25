import pandas as pd
import numpy as np

def aggregate_predictions(results):
    unique = np.unique(np.concatenate((results[0]['original'].values, results[0]['predicted'].values), axis=1))

    N_curves = len(results[0].index)
    N_trees = len(results)

    # Diccionario, cada llave es una clase con una lista de los conteos de votaciones
    # para cada curva
    result_dict = {c: np.zeros(N_curves) for c in unique}

    for i in xrange(N_curves):
        for result in results:
            result_dict[result.iloc[i]['predicted']][i] += 1

    for i in xrange(N_curves):
        for key in result_dict:
            result_dict[key][i] = result_dict[key][i] / float(N_trees)

    agg_preds = pd.DataFrame(result_dict, index=results[0].index)

    aux_dict = {'original': results[0]['original']}
    aux_dict['predicted'] = agg_preds.apply(lambda x: x.argmax(), axis=1)
    aux_dict['trust'] = agg_preds.max(axis = 1)

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
        matrix[row[0]][row[1]] += row[2]

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
