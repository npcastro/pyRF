# Toma un set de entrenamiento y genera uno con incertidumbre aleatoria en las variables
import pandas as pd
import random

# Niveles de incertidumbre que se van a generar
uncertainty_levels = [2, 6, 11, 16, 21, 26, 31, 36, 41]

# data = pd.read_csv('iris.data', sep=',', header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
data = pd.read_csv('sets/macho.txt', sep=' ', index_col=0)
del data['label']
data.rename( columns = {'label.1': 'class'}, inplace = True )
labels = {2: 'None Variable', 3: 'Quasar', 4: 'Be Stars', 5: 'Cepheid', 6: 'RR Lyrae', 7: 'Eclipsing Binaries', 8: 'MicroLensing', 9: 'Long Periodic Variable'}

aux = [labels[a] for a in data['class']]
data['class'] = pd.Series(aux, index=data.index)

# Obtengo los rangos para cada variable. Despues la incertidumbre se pone como fraccion de estos.
rangos = {col: (data[col].max() - data[col].min()) for col in data.columns[0:-1]}

for n in uncertainty_levels:

	df = {}

	for col in data.columns[0:-1]:
		feature = data[col]

		mean = []
		std = []
		l = []
		r = []

		for i in range(feature.size):
			uncertainty = random.randrange(1, n, 1) / 100.0

			valor = feature.iloc[i]
			mean.append(valor)
			std.append(rangos[col]*uncertainty/6)
			l.append(valor - (rangos[col]/2) * uncertainty)
			r.append(valor + (rangos[col]/2) * uncertainty)

		df[col + '.mean'] = mean
		df[col + '.std'] = std
		df[col + '.l'] = l
		df[col + '.r'] = r

	nuevo = pd.DataFrame(df, index = data.index)
	nuevo['weight'] = pd.Series([1.0 for i in range(len(nuevo))], index=data.index)
	nuevo['class'] = data['class']

	nuevo.to_csv('sets/macho %/macho random ' + str(n) +'.csv', index=False)