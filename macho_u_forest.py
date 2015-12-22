# coding=utf-8
# Entreno un grupo de árboles, cada uno sobre un set de entrenamiento distinto.
# Después al clasificar, junto la votación de cada árbol para tomar la decisión final

# -------------------------------------------------------------------------------------------------

from config import *
import metrics
import tree

import pandas as pd
from sklearn import cross_validation

import pickle
import sys

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        percentage = sys.argv[1]
    else:
        percentage = '100'

    folds = 4
    sets_path = '/n/seasfs03/IACS/TSC/ncastro/sets/MACHO_Sampled/' + percentage + '%/'

    arboles = []
    resultados = []
    fold_indices = []

    # Preproceso cada dataset por separado
    for i in xrange(100):
        aux_path = sets_path + 'macho_sampled_' + i + '.csv'
        data = pd.read_csv(aux_path, index_col=0)
        
        # Filtro para dejar solo las clases malas
        data = data[data['class'].apply(lambda x: True if x in ['Be_lc','EB'] else False)]

        data = data.dropna(axis=0, how='any')
        if 'weight' in data.columns.tolist():
            data = data.drop('weight', axis=1)

        y = data['class']
        data = data.drop('class', axis=1)
        data = data[['Amplitude', 'AndersonDarling', 'Autocor_length', 'Beyond1Std', 'Con', 'Eta_e', 'LinearTrend',
                     'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev', 'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude',
                     'Q31', 'Rcs', 'Skew', 'SlottedA_length', 'SmallKurtosis', 'Std', 'StetsonK', 'StetsonK_AC']]

        skf = cross_validation.StratifiedKFold(y, n_folds=folds)

        for train_index, test_index in skf:
            fold_indices.append(test_y)
            train_X, test_X = data.iloc[train_index], data.iloc[test_index]
            train_y, test_y = y.iloc[train_index], y.iloc[test_index]

            clf = None
            clf = tree.Tree('gain', max_depth=10, min_samples_split=20)

            clf.fit(train_X, train_y)
            result = clf.predict_table(test_X, test_y)
            
            arboles.append(clf)
            resultados.append(result)

            break




