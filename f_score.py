# coding=utf-8

# Toma un directorio de resultados y arma un csv con el progreso de los f_score
# -------------------------------------------------------------------------------------------------

import argparse
import sys
import os
import re

import pandas as pd

import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--path',  required=True, type=str)
parser.add_argument('--how',  required=False, default='soft', choices=['soft', 'hard'])

args = parser.parse_args(sys.argv[1:])

path = args.path
how = args.how

result_dir = path + 'Predicciones/'

p_dict = {}
r_dict = {}
f_dict = {}
w_dict = {}


files = [f for f in os.listdir(result_dir) if '.csv' in f]
for f in files:
    pattern = re.compile('[0-9]+')
    percentage = int(pattern.search(f).group())

    result = pd.read_csv(result_dir + 'result_' + str(percentage) + '.csv', index_col=0)
    if how == 'soft':
        matrix = metrics.confusion_matrix(result)
    elif how == 'hard':
        matrix = metrics.hard_matrix(result)

    matrix.to_csv(path + 'Metricas/' +  how + '_matrix_' + str(percentage) + '.csv')

    w_dict[percentage] = metrics.weighted_f_score(matrix)

    clases = matrix.columns.tolist()
    p = [metrics.precision(matrix, c) for c in clases]
    r = [metrics.recall(matrix, c) for c in clases]
    f = [metrics.f_score(matrix, c) for c in clases]

    p_dict[percentage] = p
    r_dict[percentage] = r
    f_dict[percentage] = f

save_dir = path + 'Metricas/'

w_df = pd.DataFrame.from_dict(w_dict, orient='index')
w_df.columns = ['f_score']
w_df = w_df.sort_index(ascending=True)
w_df = w_df.fillna(value=0.0)
w_df.to_csv(save_dir + how + '_weight_fscore.csv')

p_df = pd.DataFrame.from_dict(p_dict, orient='index')
p_df.columns = clases
p_df = p_df.sort_index(ascending=True)
p_df = p_df.fillna(value=0.0)
p_df.to_csv(save_dir + how + '_precision.csv')

r_df = pd.DataFrame.from_dict(r_dict, orient='index')
r_df.columns = clases
r_df = r_df.sort_index(ascending=True)
r_df = r_df.fillna(value=0.0)
r_df.to_csv(save_dir + how + '_recall.csv')

f_df = pd.DataFrame.from_dict(f_dict, orient='index')
f_df.columns = clases
f_df = f_df.sort_index(ascending=True)
f_df = f_df.fillna(value=0.0)
f_df.to_csv(save_dir + how + '_f_score.csv')
