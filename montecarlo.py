# coding=utf-8
# Entreno un grupo de árboles, cada uno sobre un set de entrenamiento distinto.
# Después al clasificar, junto la votación de cada árbol para tomar la decisión final

# -------------------------------------------------------------------------------------------------

from contextlib import closing
from functools import partial
from multiprocessing import Pool
import pickle
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
    parser.add_argument('--percentage', required=True, type=str)
    parser.add_argument('--n_processes', required=True, type=int)
    parser.add_argument('--catalog', default='MACHO', choices=['MACHO', 'EROS', 'OGLE'])
    parser.add_argument('--folds',  required=True, type=int)
    parser.add_argument('--sets_path',  required=True, type=str)
    parser.add_argument('--result_path',  required=True, type=str)
    parser.add_argument('--feature_filter',  nargs='*', type=str)

    args = parser.parse_args(sys.argv[1:])

    percentage = args.percentage
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

        print 'Entrenamiento de arboles'

        partial_train = partial(parallel.train_tree, feature_filter=feature_filter,
                                train_index=train_index)
        pool = Pool(processes=n_processes, maxtasksperchild=2)
        arboles = pool.map(partial_train, paths)
        pool.close()
        pool.join()

    #     print 'Guardado de arboles'

    #     count = 0
    #     for clf in arboles:
    #         output = open(result_path + "Arboles/arbol_" + str(count) + '.pkl', 'wb+')
    #         pickle.dump(clf, output)
    #         output.close()
    #         count += 1

        print 'Consolido resultados'

        # Guardo las votaciones de clasificaciones para cada dataset
        sample_set_result = []
        for path in paths:
            data = pd.read_csv(path, index_col=0)
            data = data.dropna(axis=0, how='any')
            y = data['class']
            data = data.drop('class', axis=1)
            if feature_filter:
                data = data[feature_filter]

            test_X = data.iloc[test_index]
            test_y = y.iloc[test_index]
            
            # Guardo la clasificacion de cada árbol para el dataset actual
            aux = []
            for clf in arboles:
                result = clf.predict_table(test_X, test_y)
                aux.append(result)

            # Consolido las votaciones de los árboles en un solo frame
            consolidated_frame = reduce(lambda a, b: a+b, map(metrics.result_to_frame, aux))
            sample_set_result.append(consolidated_frame)
            print 'Largo de lista para cada muestra: ' + str(len(sample_set_result))

        resultados.append(metrics.matrix_to_result(reduce(lambda a, b: a+b, sample_set_result), test_y))
        
        print 'Largo de lista para folds: ' + str(len(resultados))
        print 'Memoria de dataframe: ' + str(resultados[0].memory_usage(index=True))

    result = pd.concat(resultados)
    result.to_csv(result_path + 'Predicciones/result_' + percentage + '.csv')
