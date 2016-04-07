# coding=utf-8
# Entreno un grupo de árboles, cada uno sobre un set de entrenamiento distinto.
# Después al clasificar, junto la votación de cada árbol para tomar la decisión final

# -------------------------------------------------------------------------------------------------

from functools import partial
from multiprocessing import Pool
import argparse
import sys

from sklearn import cross_validation
import pandas as pd

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

    data = pd.read_csv(paths[0], index_col=0)
    y = data['class']

    skf = cross_validation.StratifiedKFold(y, n_folds=folds)

    resultados = []
    for train_index, test_index in skf:
        partial_train = partial(parallel.train_tree, feature_filter=feature_filter,
                                train_index=train_index)
        pool = Pool()
        arboles = pool.map(partial_train, paths)
        pool.close()
        pool.join()

        print 'Paso etapa de arboles'
        
        algo = []
        for path in paths:
            data = pd.read_csv(path, index_col=0)
            data = data.dropna(axis=0, how='any')
            y = data['class']
            data = data.drop('class', axis=1)
            if feature_filter:
                data = data[feature_filter]

            test_X = data.iloc[test_index]
            test_y = y.iloc[test_index]
            
            aux = []
            for clf in arboles:
                result = clf.predict_table(test_X, test_y)
                aux.append(result)
            algo.append(metrics.aggregate_predictions(aux))
            print 'Largo de lista para cada muestra: ' + str(len(algo))
        
        resultados.append(metrics.aggregate_predictions(algo))
        del aux
        del algo

        print 'Largo de lista para folds: ' + str(len(resultados))
        print 'Memoria de dataframe: ' + str(resultados[0].memory_usage(index=True))
        break

    result = pd.concat(resultados)
    result.to_csv(result_path)
