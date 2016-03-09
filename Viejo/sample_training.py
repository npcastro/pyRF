# Toma un path a un directorio con varios sets de entrenamiento
# y entrena un arbol con todos los sets que hay dentro.

# Hay que pasar solo dos posibles features, para que siempre se elijan 
# y asi poder evaluar como cambian los cortes que elige el arbol

# Despues crea los graficos con los splits inciertos

import pandas as pd
import tree
import os
import matplotlib.pyplot as plt
import numpy as np

def get_cut(node):
	return (node.feat_name, float("%.3f" % node.feat_value))

def get_split_margin( splits ):

	split_aux = []

	for split in splits:
		split_aux.append(split[1])
	
	media = np.mean(split_aux)
	std = np.std(split_aux)

	return media, media - std, media + std


def graf(data, clases, x, y, x_split, x_left, x_right, y_split, y_left, y_right, title):
	"""
	Grafica puntos y splits con incerteza.

	Parameters
	----------
	data: Dataframe con los datos a graficar
	clases: list(int) - Clases de los datos que se consideran.
	x: string - nombre de la feature del eje x
	y: string - nombre de la feature del eje y
	x_split: float - valor del split del eje x
	x_left: float - valor del margen izquierdo del split del eje x
	x_right: float - valor del margen derecho del split del eje x
	y_split: float - valor del split del eje y
	y_left: float - valor del margen izquierdo del split del eje y
	y_right: float - valor del margen derecho del split del eje y
	title: titulo del grafico
	"""
	
	# Filtro las clases
	criterion = data['class'].map(lambda x: x in clases)
	data = data[criterion]

	colors = { 'Be Stars': 'b', 'Cepheid': 'g', 'Eclipsing Binaries': 'r', 'Long Periodic Variable': 'c', 'MicroLensing': 'm', 'None Variable': 'y', 'Quasar': 'k', 'RR Lyrae':'w' }

	# Encontrar margenes del grafico
	# x_mean = data[x + '.mean'].tolist()	
	# y_mean = data[y + '.mean'].tolist()
	# plt.xlim(min(x_mean) - max(x_margin), max(x_mean) + max(x_margin))
	# plt.ylim(min(y_mean) - max(y_margin), max(y_mean) + max(y_margin))

	fig = plt.figure(figsize=(6*3.13,9))
	ax = fig.add_subplot(111)

	# Graficar
	for c in clases:
		x_mean = data[data['class'] == c][x + '.mean'].tolist()
		y_mean = data[data['class'] == c][y + '.mean'].tolist()
		ax.scatter( x_mean, y_mean, c = colors[c] )	

	plt.xlabel(x)
	plt.ylabel(y, rotation = 'horizontal')

	plt.axhline(y = y_split, color = 'r')
	plt.axvline(x = x_split, color = 'r')

	# Margen
	x_min, x_max = plt.xlim()
	y_min, y_max = plt.ylim()

	plt.fill_between([x_left, x_right], y_min, y_max, facecolor='gray', alpha=0.5)
	plt.fill_between( [x_min, x_max], y_left, y_right, facecolor='gray', alpha=0.5)

	# plt.xlim(x_min, x_max)
	# plt.ylim(y_min, y_max)
	plt.xlim(-0.2, 0.6)
	plt.ylim(-0.4, 1.0)

	plt.title(title)
	

	plt.show()
	# plt.savefig('Resultados/' + title)
	plt.close()

if __name__ == '__main__':
	# aux = [1,5,10,15,20,25,30,35,40]
	aux = [5]
	for n in aux:

		# directory = 'sets/sampling/1 %'
		directory = 'sets/sampling/' + str(n) +' %'

		x = 'N_below_4_B'
		y = 'Period_SNR_R'

		for root, dirs, files in os.walk(directory):
			training_sets_paths = [directory + '/' + f for f in files if not f[0] == '.']

		# Entreno un modelo para cada set sampleado y guardo sus cortes
		root_cuts, right_cuts, left_cuts = [], [], []
		for path in training_sets_paths:

			data = pd.read_csv(path)

			# Elijo solo algunas clases
			clases = [ 'Long Periodic Variable', 'None Variable', 'RR Lyrae']
			criterion = data['class'].map(lambda x: x in clases)
			data = data[criterion]

			# Dejo solo algunas features
			columns = ['N_below_4_B.mean', 'Period_SNR_R.mean', 'class']
			data = data[columns]

			# Entreno arbol
			clf = tree.Tree('gain', max_depth=3)
			clf.fit(data)

			root_cuts.append( get_cut(clf.root))
			left_cuts.append( get_cut(clf.root.left))
			right_cuts.append( get_cut(clf.root.right))

		# Obtengo el margen de los cortes
		x_split, x_left, x_right = get_split_margin(right_cuts)
		y_split, y_left, y_right = get_split_margin(root_cuts)

		# Grafico
		graf(data, clases, x, y, x_split, x_left, x_right, y_split, y_left, y_right, 'Tree ' + x + ' vs ' + y + ' ' +str(n) + '%.png')
