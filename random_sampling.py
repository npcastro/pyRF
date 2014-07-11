# Toma un dataset con incertidumbre y genera n datasets nuervos sampleados de el. 

import pandas as pd
from numpy import random

def filterfeatures(df):
	"""Toma un dataframe y retorna el nombre de las features.
	"""
	filter_arr = []
	for f in df.columns:
		if not '.l' in f and not '.r' in f and not '.std' in f and f != 'weight' and f != 'class':
			filter_arr.append(f.rstrip('.mean'))
	return filter_arr

def sample_row(row, columns):
	"""Recibe una fila de datos con incertidumbre gaussiana y retorna un dato sampleado aleatoriamente de esta.
	"""
	sampled_row = pd.Series( index = columns)

	# Sampleo cada feature segun la distribucion de la fila
	for c in columns:
		sampled_row[c] = random.normal(row[c + '.mean'], row[c + '.std'])

	# Agrego la columna clase
	sampled_row['class'] = row['class']

	return sampled_row


if __name__ == '__main__':
	
	path = 'sets/macho random sampled 10 folds.csv'
	data = pd.read_csv(path)
	
	n = 15
	for i in xrange(n):

		columns = filterfeatures(data)
		sampled_frame = data.apply(sample_row, axis=1, args=[columns])

		sampled_frame.to_csv('sets/sampling/sampled macho ' + str(i) + '.csv')


