# Guardo los metodos basicos para graficar y mantener el codigo dry

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import pandas as pd
import tree

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
	# path = 'sets/artificial %/artificial random 11.csv'
	# data = pd.read_csv(path)

	colors = {'blue': 'b', 'red': 'r'}

	###### Entreno arbol clasico ######
	# clf_clasico = tree.Tree('gain', max_depth=3)
	# clf_clasico.fit(data)

	# root_split_name = clf_clasico.root.feat_name.rstrip('.mean')
	# root_split_value = clf_clasico.root.feat_value

	# right_split_name = clf_clasico.root.right.feat_name.rstrip('.mean')
	# right_split_value = clf_clasico.root.right.feat_value

	# ###### Grafico cortes #######
	# fig = plt.figure(1, figsize=(6*3.13,9))
	# ax = fig.add_subplot(111)

	# draw_points(ax, data, root_split_name, right_split_name, colors)

	# draw_splits(ax, root_split_value, right_split_value, 'r')
	# add_names('Feature 1 v/s 2', root_split_name, right_split_name)

	# plt.show()
	# # plt.savefig('Resultados/F1 vs F2')
	# plt.close()

	###### Con incertidumbre en una variable ######

	for u in [2, 6, 11, 16, 21, 26, 31, 36, 41]:
		path = 'sets/artificial %/artificial random ' + str(u) + '.csv'
		data = pd.read_csv(path)

		fig = plt.figure(1, figsize=(6*3.13,9))
		ax = fig.add_subplot(111)
		
		# Elimino incertidumbre de una dimension
		data['Feature 2.std'] = 0.0000001

		# Entreno arbol incierto
		clf = tree.Tree('uncertainty', min_samples_split = 50, most_mass_threshold=0.99999, min_mass_threshold=0.00001, min_weight_threshold=0.00001, max_depth=3)
		clf.fit(data)

		# Pongo un nivel de icnertidumbre arbitrario para que se vea bien el grafico
		data['Feature 2.std'] = 0.01

		root_split_name = clf.root.feat_name.rstrip('.mean')
		root_split_value = clf.root.feat_value
		right_split_name = clf.root.right.feat_name.rstrip('.mean')
		right_split_value = clf.root.right.feat_value

		draw_ellipses(ax, data, root_split_name, right_split_name, colors)
		# set_ellipses_graph_lims(data,root_split_name, right_split_name)
		plt.xlim(-0.5,6)
		plt.ylim(-0.5,6)
		draw_splits(ax, root_split_value, right_split_value, 'g')
		title = root_split_name + ' vs ' + right_split_name + ' ' + str(u - 1) + '%'
		add_names(title, root_split_name, right_split_name)

		# plt.show()
		plt.savefig('Resultados/' + title)
		plt.close()

