# coding=utf-8
# Entra un arbol de decisión con incertidumbre en paralelo
# Y guarda sus resultados


import tree

import pandas
from sklearn import cross_validation

import pickle
import sys

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        percentage = sys.argv[1]
    else:
        percentage = '100'

    folds = 10
    training_set_path = '' 

