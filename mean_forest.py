# coding=utf-8

# Entreno un grupo de árboles, cada uno sobre un set de entrenamiento distinto.
# Después ocupo la media de cada curva para clasificar y junto la votación de cada árbol para tomar
# la decisión final

# -------------------------------------------------------------------------------------------------

from multiprocessing import Pool
from functools import partial
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
    parser.add_argument('--test_path',  required=True, type=str)
    parser.add_argument('--result_path',  required=True, type=str)
    parser.add_argument('--feature_filter',  nargs='*', type=str)

    args = parser.parse_args(sys.argv[1:])

    n_processes = args.n_processes
    catalog = args.catalog
    folds = args.folds
    sets_path = args.sets_path
    test_path = args.test_path
    result_path = args.result_path
    feature_filter = args.feature_filter

    paths = [sets_path + catalog + '_sampled_' + str(i) + '.csv' for i in xrange(100)]

    partial_fit = partial(parallel.fit_means_tree, test_path, feature_filter=feature_filter, folds=10)
    
    pool = Pool(processes=n_processes, maxtasksperchild=2)
    resultados = pool.map(partial_fit, paths)
    pool.close()
    pool.join()

    result = metrics.aggregate_predictions(resultados)
    result.to_csv(result_path)
