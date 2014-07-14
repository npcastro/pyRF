# Guardo los metodos basicos para graficar y mantener el codigo dry

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import pandas as pd

def draw_points( ax, df, x_name, y_name, colors):
	"""Toma un dataframe y grafica dos variables como puntos con distintos colores

	ax: objeto axis de pyplot
	df: dataframe con los datos a graficar. (Los nombres de las features deben venir con .mean)
	x_name: nombre de la variable x
	y_name: nombre de la variable y
	colors: dict con los colores correspondientes a cada clase
	"""

	if 'weight' in df.columns:
		aux = '.mean'
	else: 
		aux = ''

	for c in df['class'].unique().tolist():
		x = df[df['class'] == c][x_name + aux].tolist()
		y = df[df['class'] == c][y_name + aux].tolist()
		ax.scatter( x, y, c = colors[c] )

def draw_error_bars(ax, data, x_name, y_name, colors):
	# Obtengo los puntos y sus incertidumbres

	for c in data['class'].unique().tolist():
		x_mean = data[data['class'] == c][x_name + '.mean'].tolist()
		x_margin = (data[data['class'] == c][x_name + '.std'] * 6).tolist()
		y_mean = data[data['class'] == c][y_name + '.mean'].tolist()
		y_margin = (data[data['class'] == c][y_name + '.std'] * 6).tolist()

		ax.errorbar(x_mean, y_mean, y_margin, x_margin, ls='none', color=colors[c], marker='o')

def draw_ellipses(ax, data, x_name, y_name, colors):

	# Obtengo los valores de las elipses
	x_mean = data[x_name + '.mean'].tolist()
	x_margin = (data[x_name + '.std'] * 6).tolist()
	y_mean = data[y_name + '.mean'].tolist()
	y_margin = (data[y_name + '.std'] * 6).tolist()

	class_list = data['class'].tolist()

	ells = []
	for i in xrange(len(x_mean)):
		ells.append( Ellipse(xy=[x_mean[i], y_mean[i]], width=x_margin[i], height=y_margin[i], angle=0, facecolor=colors[class_list[i]]) )

	for e in ells:
	    ax.add_artist(e)
	    e.set_clip_box(ax.bbox)
	    e.set_alpha(0.7)

def set_ellipses_graph_lims(data, x_name, y_name):
	
	# Obtengo los valores de las elipses
	x_mean = data[x_name + '.mean'].tolist()
	x_margin = (data[x_name + '.std'] * 6).tolist()
	y_mean = data[y_name + '.mean'].tolist()
	y_margin = (data[y_name + '.std'] * 6).tolist()

	plt.xlim(min(x_mean) - max(x_margin), max(x_mean) + max(x_margin))
	plt.ylim(min(y_mean) - max(y_margin), max(y_mean) + max(y_margin))
	

def draw_splits( ax, x, y, color):
	ax.axvline(x = x, color=color)
	ax.axhline(y = y, color =color)

def add_names( title, x_name, y_name):
	plt.title(title)
	plt.xlabel(x_name)
	plt.ylabel(y_name, rotation = 'horizontal', labelpad=20)
	

if __name__ == '__main__':
	path = 'sets/artificial random 10.csv'
	data = pd.read_csv(path)

	colors = {'blue': 'b', 'red': 'r'}

	fig = plt.figure(1, figsize=(6*3.13,9))
	ax = fig.add_subplot(111)

	draw_points(ax, data, 'Feature 1', 'Feature 2', colors)
	draw_splits(ax, 2.65, 1.95, 'r')
	add_names('Feature 1 v/s 2', 'Feature 1', 'Feature 2')

	# plt.show()
	# plt.savefig('Resultados/F1 vs F2')
	plt.close()

	###### Con incertidumbre en una variable ######

	fig = plt.figure(1, figsize=(6*3.13,9))
	ax = fig.add_subplot(111)
	
	# Elimino incertidumbre de una dimension
	# data['Feature 2.std'] = 0.0000001
	data['Feature 2.std'] = 0.005

	# Entreno arbol incierto
	# import tree
	# clf = tree.Tree('uncertainty', min_samples_split = 50, most_mass_threshold=0.99999, min_mass_threshold=0.00001, min_weight_threshold=0.00001, max_depth=3)
	# clf.fit(data)

	# draw_error_bars(ax, data, 'Feature 1', 'Feature 2', colors)

	draw_ellipses(ax, data, 'Feature 1', 'Feature 2', colors)
	set_ellipses_graph_lims(data,'Feature 1', 'Feature 2')
	draw_splits(ax, 2.65, 1.95, 'k')
	add_names('Feature 1 v/s 2', 'Feature 1', 'Feature 2')

	plt.show()

