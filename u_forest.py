# coding=utf-8
# Entreno un grupo de árboles, cada uno sobre un set de entrenamiento distinto.
# Después al clasificar, junto la votación de cada árbol para tomar la decisión final

# -------------------------------------------------------------------------------------------------

from functools import partial
from multiprocessing import Pool
import argparse
import sys

from config import *
import metrics
import parallel

if __name__ == '__main__':

    # Recibo parámetros de la linea de comandos
    print ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_processes', required=True, type=int)
    parser.add_argument('--catalog', default='MACHO', choices=['MACHO', 'EROS', 'OGLE'])
    parser.add_argument('--folds',  required=True, type=int)
    parser.add_argument('--sets_path',  required=True, type=str)
    parser.add_argument('--result_path',  required=True, type=str)
    parser.add_argument('--feature_filter',  nargs='*', type=str)

    args = parser.parse_args(sys.argv[1:])

    catalog = args.catalog
    n_processes = args.n_processes
    folds = args.folds
    sets_path = args.sets_path
    result_path = args.result_path
    feature_filter = args.feature_filter

    paths = [sets_path + catalog + '_sampled_' + str(i) + '.csv' for i in xrange(100)]
    
    # Para asegurar que sean las mismas curvas que en el caso normal
    # index_filter = pd.read_csv('/n/home09/ncastro/workspace/Features/sets/MACHO_Reduced/Macho reduced set '
    #                      + percentage + '.csv', index_col=0).index

    # class_filter = ['Be_lc','EB']

    feature_filter = ['Amplitude', 'AndersonDarling', 'Autocor_length', 'Beyond1Std', 'Con',
                      'Eta_e', 'LinearTrend', 'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
                      'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'Q31', 'Rcs', 'Skew',
                      'SlottedA_length', 'SmallKurtosis', 'Std', 'StetsonK', 'StetsonK_AC']

    partial_fit = partial(parallel.fit_tree, feature_filter=feature_filter, folds=folds)
    pool = Pool()
    
    resultados = pool.map(partial_fit, paths)
    pool.close()
    pool.join()

    result = metrics.aggregate_predictions(resultados)
    result.to_csv(result_path)
