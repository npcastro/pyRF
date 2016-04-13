# coding=utf-8

# Voy a recorrer uno a uno los sets sampleados, entrenar un arbol de decision normal y un RF,
# y a comparar el f_score de cada uno sobre ellos. En teoría el rf deberia ser mejor que
# el arbol de decision en la mayoría de los casos
# -------------------------------------------------------------------------------------------------

from multiprocessing import Pool
from functools import partial
import argparse
import sys

import metrics
import parallel

if __name__ == '__main__':

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
    paths = paths[0:10]

    # Entreno y clasifico con árboles
    partial_fit = partial(parallel.fit_tree, feature_filter=feature_filter, folds=folds)
    pool = Pool(processes=n_processes, maxtasksperchild=2)
    resultados_tree = pool.map(partial_fit, paths)
    pool.close()
    pool.join()

    # Imprimo y guardo resultados obtenidos
    for i, r in resultados_tree:
        r.to_csv(result_path + 'result_tree_' + str(i) + '.csv')
        matrix = metrics.hard_matrix(r)
        print 'Tree ' + str(i) + ' f_score: ' + str(metrics.weighted_f_score(matrix))

    # Entreno y clasifico con rf
    partial_fit = partial(parallel.fit_rf, feature_filter=feature_filter, folds=folds)
    pool = Pool(processes=n_processes, maxtasksperchild=2)
    resultados_rf = pool.map(partial_fit, paths)
    pool.close()
    pool.join()

    # Imprimo y guardo resultados obtenidos
    for i, r in resultados_tree:
        r.to_csv(result_path + 'result_rf_' + str(i) + '.csv')
        matrix = metrics.hard_matrix(r)
        print 'RF ' + str(i) + ' f_score: ' + str(metrics.weighted_f_score(matrix))