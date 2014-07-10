# Script para hacer los graficos que comparan los cortes del arbol con incertidumbre versus el arbol clasico

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse

def draw_ellipses(ax, x_mean, y_mean, x_margin, y_margin, class_list, colors):
	ells = []
	for i in xrange(len(x_mean)):
		ells.append( Ellipse(xy=[x_mean[i], y_mean[i]], width=x_margin[i], height=y_margin[i], angle=0, facecolor=colors[class_list[i]]) )

	for e in ells:
	    ax.add_artist(e)
	    e.set_clip_box(ax.bbox)
	    e.set_alpha(0.7)

def draw_points():
	pass

def graf(path, clases, x, y, x_value, y_value, how='points', show=True):
	""" Grafica datos con incertidumbre como elipses en el espacio.

	Parameters
	----------
	path: string
		El path al set de datos que se va a graficar.
	clases: list(str)
		Nombres de las clases que se van a considerar
	x: string
		nombre de la feature en el eje x
	y: string
		nombre de la feature en el eje y
	x_value: float
		valor donde se grafica una linea vertical
	y_value: float
		valor donde se grafica una linea horizontal
	"""
	data = pd.read_csv(path)

	# Dejo solo las clases objetivo
	criterion = data['class'].map(lambda x: x in clases)
	data = data[criterion]

	# Obtengo los valores de las elipses
	x_mean = data[x + '.mean'].tolist()
	x_margin = (data[x + '.std'] * 6).tolist()
	y_mean = data[y + '.mean'].tolist()
	y_margin = (data[y + '.std'] * 6).tolist()

	# Colores para las distinas clases
	colors = { 'Be Stars': 'b', 'Cepheid': 'g', 'Eclipsing Binaries': 'r', 'Long Periodic Variable': 'c', 'MicroLensing': 'm', 'None Variable': 'y', 'Quasar': 'k', 'RR Lyrae':'w' }
	class_list = data['class'].tolist()

	# Defino los margenes para el grafico
	plt.xlim(min(x_mean) - max(x_margin), max(x_mean) + max(x_margin))
	plt.ylim(min(y_mean) - max(y_margin), max(y_mean) + max(y_margin))

	fig = plt.figure(1)
	ax = fig.add_subplot(111)

	if how == 'ellipses':
		draw_ellipses(ax, x_mean, y_mean, x_margin, y_margin, class_list, colors)

	plt.xlabel(x)
	plt.ylabel(y, rotation = 'horizontal')

	plt.axhline(y = y_value, color = 'r')
	plt.axvline(x = x_value, color = 'r')

	plt.title('utree ' + x + ' vs ' + y + '.png')

	if show:
		plt.show()
	else:
		plt.savefig('Resultados/utree ' + x + ' vs ' + y + '.png')

	plt.close()

def graf_points( path, clases, x, y, x_value, y_value ):
	
	data = pd.read_csv(path)
	
	# Elijo solo algunas clases
	criterion = data['class'].map(lambda x: x in clases)
	data = data[criterion]

	colors = { 'Be Stars': 'b', 'Cepheid': 'g', 'Eclipsing Binaries': 'r', 'Long Periodic Variable': 'c', 'MicroLensing': 'm', 'None Variable': 'y', 'Quasar': 'k', 'RR Lyrae':'w' }

	# Encontrar margenes del grafico
	x_mean = data[x + '.mean'].tolist()
	x_margin = (data[x + '.std'] * 6).tolist()
	y_mean = data[y + '.mean'].tolist()
	y_margin = (data[y + '.std'] * 6).tolist()

	plt.xlim(min(x_mean) - max(x_margin), max(x_mean) + max(x_margin))
	plt.ylim(min(y_mean) - max(y_margin), max(y_mean) + max(y_margin))

	fig = plt.figure(1)
	ax = fig.add_subplot(111)

	# Graficar
	for c in clases:

		x_mean = data[data['class'] == c][x + '.mean'].tolist()
		x_margin = ((data[x + '.r'] - data[x + '.l'])/1).tolist()
		y_mean = data[data['class'] == c][y + '.mean'].tolist()
		y_margin = ((data[y + '.r'] - data[y + '.l'])/1).tolist()

		ax.scatter( x_mean, y_mean, c = colors[c] )	

	plt.xlabel(x)
	plt.ylabel(y, rotation = 'horizontal')

	plt.axhline(y = y_value, color = 'r')
	plt.axvline(x = x_value, color = 'r')

	plt.title('tree ' + x + ' vs ' + y + '.png')
	# plt.show()
	plt.savefig('Resultados/tree ' + x + ' vs ' + y + '.png')
	plt.close()

def graf_points_and_ellipses( path, clases, x, y, x_value, y_value ):
	
	data = pd.read_csv(path)
	
	# Elijo solo algunas clases
	criterion = data['class'].map(lambda x: x in clases)
	data = data[criterion]

	colors = { 'Be Stars': 'b', 'Cepheid': 'g', 'Eclipsing Binaries': 'r', 'Long Periodic Variable': 'c', 'MicroLensing': 'm', 'None Variable': 'y', 'Quasar': 'k', 'RR Lyrae':'w' }

	x_mean = data[x + '.mean'].tolist()
	x_margin = (data[x + '.std'] * 6).tolist()
	y_mean = data[y + '.mean'].tolist()
	y_margin = (data[y + '.std'] * 6).tolist()

	plt.xlim(min(x_mean) - max(x_margin), max(x_mean) + max(x_margin))
	plt.ylim(min(y_mean) - max(y_margin), max(y_mean) + max(y_margin))

	class_list = data['class'].tolist()

	fig = plt.figure(1)
	ax = fig.add_subplot(111)

	# Grafico elipses
	ells = []
	for i in xrange(len(x_mean)):
		ells.append( Ellipse(xy=[x_mean[i], y_mean[i]], width=x_margin[i], height=y_margin[i], angle=0, facecolor=colors[class_list[i]]) )

	for e in ells:
	    ax.add_artist(e)
	    e.set_clip_box(ax.bbox)
	    e.set_alpha(0.7)	

	# Grafico puntos
	for c in ['None Variable', 'Long Periodic Variable']:
		x_mean = data[data['class'] == c][x + '.mean'].tolist()
		x_margin = (data[x + '.std'] * 6).tolist()
		y_mean = data[data['class'] == c][y + '.mean'].tolist()
		y_margin = (data[y + '.std'] * 6).tolist()

		ax.scatter( x_mean, y_mean, c = colors[c], alpha=0.7, zorder=100 )

	plt.xlabel(x)
	plt.ylabel(y, rotation = 'horizontal')

	plt.axhline(y = y_value, color = 'r')
	plt.axvline(x = x_value, color = 'r')

	plt.title('utree ' + x + ' vs ' + y + '.png')
	# plt.show()
	plt.savefig('Resultados/Utree ' + x + ' vs ' + y + '.png')
	plt.close()

if __name__ == '__main__':
	path = 'sets/macho random sampled 10 folds.csv'
	# path = 'sets/macho std variable.csv'
	# path = 'sets/macho artificial final.csv'

	# clases = [ 'Long Periodic Variable', 'MicroLensing', 'None Variable', 'RR Lyrae']
	clases = [ 'Long Periodic Variable', 'None Variable', 'RR Lyrae']


	x = 'CAR_mean_R'
	y = 'N_below_4_B'
	z = 'Period_SNR_R'

	###### Cortes arbol clasico ######

	x_value = -7.42
	y_value = 0.05
	z_value = 0.05

	# graf_points(path, clases, x, y, x_value, y_value)
	# graf_points(path, clases, x, z, x_value, z_value)
	# graf_points(path, clases, y, z, y_value, z_value)

	###### Cortes arbol incierto ######

	x_value = -7.42
	y_value = 0.03
	z_value = 0.04

	# graf(path, clases, x, y, x_value, y_value, how='ellipses')
	# graf(path, clases, x, z, x_value, z_value, how='ellipses')
	# graf(path, clases, y, z, y_value, z_value, how='ellipses')


	###### Cortes inciertos y puntos ######
	# graf_points_and_ellipses(path, clases, y, z, y_value, z_value)

	