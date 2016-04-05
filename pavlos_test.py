# coding=utf-8
# Entreno un modelo (random forest) sobre un set de entrenamiento normal
# Después al testear testeo con las medias del random forest, en lugar de con las feats normales

# -------------------------------------------------------------------------------------------------

import argparse
import sys

import pandas as pd

print ' '.join(sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument('--catalog', default='MACHO', choices=['MACHO', 'EROS', 'OGLE'])
parser.add_argument('--folds',  required=True, type=int)
parser.add_argument('--regular_set_path',  required=True, type=str)
parser.add_argument('--mean_set_path',  required=True, type=str)
parser.add_argument('--result_path',  required=True, type=str)
parser.add_argument('--feature_filter',  nargs='*', type=str)

args = parser.parse_args(sys.argv[1:])

catalog = args.catalog
folds = args.folds
regular_set_path = args.regular_set_path
mean_set_path = args.mean_set_path
result_path = args.result_path
feature_filter = args.feature_filter

# Necesito asgurarme de tener los mismos ids en ambos sets. Normal y con medias
train_data = pd.read_csv(regular_set_path, index_col=0)
test_data = pd.read_csv(mean_set_path, index_col=0)

