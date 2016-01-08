# coding=utf-8
# Entreno un grupo de árboles, cada uno sobre un set de entrenamiento distinto.
# Después ocupo la media de cada curva para clasificar y junto la votación de cada árbol para tomar
# la decisión final

# -------------------------------------------------------------------------------------------------

from config import *
import metrics
import parallel

import sys

from functools import partial
from multiprocessing import Pool

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        percentage = sys.argv[1]
    else:
        percentage = '100'

    folds = 10

    sets_path = '/n/seasfs03/IACS/TSC/ncastro/sets/MACHO_Sampled/' + percentage + '%/'
    paths = [sets_path + 'macho_sampled_' + str(i) + '.csv' for i in xrange(100)]

    test_path = '/n/home09/ncastro/workspace/Features/sets/MACHO_Means/Macho means set ' + percentage + '.csv'
    
    # Para asegurar que sean las mismas curvas que en el caso normal
    # index_filter = pd.read_csv('/n/home09/ncastro/workspace/Features/sets/MACHO_Reduced/Macho reduced set '
    #                      + percentage + '.csv', index_col=0).index

    # class_filter = ['Be_lc','EB']

    feature_filter = ['Amplitude', 'AndersonDarling', 'Autocor_length', 'Beyond1Std', 'Con',
                      'Eta_e', 'LinearTrend', 'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
                      'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'Q31', 'Rcs', 'Skew',
                      'SlottedA_length', 'SmallKurtosis', 'Std', 'StetsonK', 'StetsonK_AC']

    partial_fit = partial(parallel.fit_means_tree, test_path, feature_filter=feature_filter, folds=10)
    pool = Pool()
    
    # resultados = pool.map(partial_fit, paths)
    resultados = map(partial_fit, paths)
    pool.close()
    pool.join()

    result = metrics.aggregate_predictions(resultados)

    result.to_csv('/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Sampled/Means/Predicciones/result_' + percentage + '.csv')
