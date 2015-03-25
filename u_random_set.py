# Toma un set de entrenamiento y genera uno con incertidumbre aleatoria en las variables

import pandas as pd
import random
import numpy as np

def add_uncertainty(data, level):
	"""Toma un dataframe normal (cada columna es una feature y la ultima es class), y le agrega incertidumbre

	data: dataframe
	level: porcentaje del rango. Se usa como el valor maximo de incertidumbre que se le da a un punto
	"""
	
	# Obtengo los rangos para cada variable. Despues la incertidumbre se pone como fraccion de estos.
	rangos = {col: (data[col].max() - data[col].min()) for col in data.columns[0:-1]}

	df = {}

	for col in data.columns[0:-1]:
		feature = data[col]

		mean = []
		std = []
		l = []
		r = []

		for i in range(feature.size):
			uncertainty = random.randrange(1, level, 1) / 100.0

			sampled_value = np.random.normal(feature.iloc[i], rangos[col]*uncertainty/6)
			mean.append(sampled_value)
			std.append(rangos[col]*uncertainty/6)
			l.append(sampled_value - (rangos[col]/2) * uncertainty)
			r.append(sampled_value + (rangos[col]/2) * uncertainty)

		df[col + '.mean'] = mean
		df[col + '.std'] = std
		df[col + '.l'] = l
		df[col + '.r'] = r

	nuevo = pd.DataFrame(df, index = data.index)
	nuevo['weight'] = pd.Series([1.0 for i in range(len(nuevo))], index=data.index)
	nuevo['class'] = data['class']

	return nuevo


if __name__ == '__main__':

	data = pd.read_csv('sets/Macho.csv', index_col = 0)
	uncertainty_levels = range(5, 70, 5)

	for u in uncertainty_levels:	
		u_data = add_uncertainty(data, u)
		u_data.to_csv('sets/Macho random/Macho random ' + str(u) +'.csv', index=False)


# data = pd.read_csv('iris.data', sep=',', header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

# Para leer bien macho.txt
# data = pd.read_csv('sets/macho.txt', sep=' ', index_col=0)
# del data['label']
# data.rename( columns = {'label.1': 'class'}, inplace = True )
# labels = {2: 'None Variable', 3: 'Quasar', 4: 'Be Stars', 5: 'Cepheid', 6: 'RR Lyrae', 7: 'Eclipsing Binaries', 8: 'MicroLensing', 9: 'Long Periodic Variable'}
# aux = [labels[a] for a in data['class']]
# data['class'] = pd.Series(aux, index=data.index)
