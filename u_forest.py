# coding=utf-8
# Entreno un grupo de árboles, cada uno sobre un set de entrenamiento distinto.
# Después al clasificar, junto la votación de cada árbol para tomar la decisión final

# -------------------------------------------------------------------------------------------------

from functools import partial
from multiprocessing import Pool
import argparse
import sys

import pandas as pd

import metrics
import parallel

if __name__ == '__main__':

    # Recibo parámetros de la linea de comandos
    print ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_processes', required=True, type=int)
    parser.add_argument('--n_samples', required=True, type=int)
    parser.add_argument('--catalog', default='MACHO', choices=['MACHO', 'EROS', 'OGLE'])
    parser.add_argument('--folds',  required=True, type=int)
    parser.add_argument('--model', default='tree', choices=['tree', 'rf', 'sktree'])
    parser.add_argument('--inverse', required=False, action='store_true')

    parser.add_argument('--max_depth',  required=False, type=int)
    parser.add_argument('--min_samples_split',  required=False, type=int)

    parser.add_argument('--sets_path',  required=True, type=str)
    parser.add_argument('--result_path',  required=True, type=str)
    
    parser.add_argument('--lc_filter', required=False, type=float, 
                        help='Percentage of the total amount of data to use')
    parser.add_argument('--index_filter', required=False, type=str)
    parser.add_argument('--feature_filter',  nargs='*', type=str)

    args = parser.parse_args(sys.argv[1:])

    catalog = args.catalog
    n_processes = args.n_processes
    n_samples = args.n_samples
    folds = args.folds
    model = args.model
    inverse = args.inverse

    max_depth = args.max_depth
    min_samples_split = args.min_samples_split

    sets_path = args.sets_path
    result_path = args.result_path
    
    lc_filter = args.lc_filter
    index_filter = args.index_filter
    feature_filter = args.feature_filter

    if index_filter is not None:
        index_filter = pd.read_csv(index_filter, index_col=0).index

    paths = [sets_path + catalog + '_sampled_' + str(i) + '.csv' for i in xrange(n_samples)]

    if model == 'tree':
        partial_fit = partial(parallel.fit_tree, feature_filter=feature_filter, folds=folds,
                              inverse=inverse, max_depth=max_depth,
                              min_samples_split=min_samples_split, lc_filter=lc_filter)
    elif model == 'rf':
        partial_fit = partial(parallel.fit_rf, feature_filter=feature_filter, folds=folds,
                              inverse=inverse, lc_filter=lc_filter)
    elif model == 'sktree':
        partial_fit = partial(parallel.fit_sktree, feature_filter=feature_filter, folds=folds,
                              inverse=inverse, max_depth=max_depth,
                              min_samples_split=min_samples_split, lc_filter=lc_filter)

    pool = Pool(processes=n_processes, maxtasksperchild=2)
    
    resultados = pool.map(partial_fit, paths)
    # resultados = map(partial_fit, paths)
    pool.close()
    pool.join()

    result = metrics.aggregate_predictions(resultados)
    result.to_csv(result_path)

    print metrics.weighted_f_score(metrics.confusion_matrix(result))
