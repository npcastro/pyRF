# Este script toma un dataset entrena un arbol con el 
# y guarda todos los splits que realiza.

import pandas as pd
import tree
import pickle
import numpy as np
import random

if __name__ == '__main__':

    data = pd.read_csv('sets/Macho.csv', index_col = 0)
    data = data.dropna(axis=0, how='any')

    y = data['class']
    data = data.drop('class', axis=1)

    clf = tree.Tree('gain')
    clf.fit(data, y)

    splits = clf.get_splits()

    output = open('Macho splits.pkl', 'w')
    pickle.dump(splits, output)
    output.close()
