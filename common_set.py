# coding=utf-8

# Codigo para separar catalogos en sets de entrenamiento, 
# con distinto tamaño y un set de testing único. 

# La idea es siempre mantener las proporciones
# Y preocuparme de que las curvas de training y testing se encuentren tanto
# Las curvas normales como en las sampleadas

# -----------------------------------------------------------------------------

import argparse
import sys

from sklearn import cross_validation
import pandas as pd

if __name__ == '__main__':

    print ' '.join(sys.argv)
    parser = argparse.ArgumentParser()

    parser.add_argument('--catalog', default='MACHO', choices=['MACHO', 'EROS', 'OGLE'])
    parser.add_argument('--training_size', required=True, type=int)

    args = parser.parse_args(sys.argv[1:])

    catalog = args.catalog
    training_size = args.training_size
    
    common_set = pd.read_csv('/n/home09/ncastro/workspace/Features/sets/Common/' + catalog + '.csv', index_col=0)
    test_set = pd.read_csv('/n/home09/ncastro/workspace/Features/sets/Common/' + catalog + '_test.csv', index_col=0)

    train_set = common_set.loc[common_set.index.difference(test_set.index)]

    sss = cross_validation.StratifiedShuffleSplit(train_set['class'], n_iter=1,
                                                  train_size=training_size, test_size=None,
                                                  random_state=1)

    for train_index, test_index in sss:
        filtered_df = train_set.iloc[train_index]

    filtered_df.to_csv('/n/home09/ncastro/workspace/Features/sets/Common/' + catalog + '_' +
                       str(training_size) + '.csv')