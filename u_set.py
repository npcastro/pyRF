import pandas as pd

# data = pd.read_csv('sets/iris.data', sep=',', header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
data = pd.read_csv('sets/macho.txt', sep=' ', index_col=0)
del data['label']
data.rename( columns = {'label.1': 'class'}, inplace = True )
labels = {2: 'None Variable', 3: 'Quasar', 4: 'Be Stars', 5: 'Cepheid', 6: 'RR Lyrae', 7: 'Eclipsing Binaries', 8: 'MicroLensing', 9: 'Long Periodic Variable'}

aux = [labels[a] for a in data['class']]
data['class'] = pd.Series(aux, index=data.index)

porcentajes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for u in porcentajes:

	uncertainty = u

	# Calculo el rango que toma cada feature en el training set. Estos valores me serviran para definir los 
	# rangos de valores que podran tomar los valores cuando le agregue la incertidumbre al set. 
	rangos = {col: (data[col].max() - data[col].min()) for col in data.columns[0:-1]}


	# Calculo las medias, desviaciones y dominios para todas las tuplas del frame. 
	# Asumo que la desviacion estandar es un sexto del rango. De esta forma, obtengo una gaussiana con un 99% de la masa
	# de probabilidad. 
	df = {}
	# Recorro cada columna salvo la clase
	for col in data.columns[0:-1]:
		feature = data[col]
		aux = []

		mean = []
		std = []
		l = []
		r = []

		# para cada valor de la feature, excepto la clase
		for i in range(feature.size):
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

	# nuevo.to_csv('sets/u_iris ' + str(int(uncertainty*100)) +'.csv', index=False)
	nuevo.to_csv('sets/macho ' + str(int(uncertainty*100)) +'.csv', index=False)