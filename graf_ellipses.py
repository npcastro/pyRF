# Script para hacer los graficos que comparan los cortes del arbol con incertidumbre versus el arbol clasico

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse


def graf_ellipses(path, clases, x, y, x_value, y_value):
	data = pd.read_csv(path)

	# Elijo solo algunas clases
	criterion = data['class'].map(lambda x: x in clases)
	data = data[criterion]

	# obtengo los valores de las elipses
	x_mean = data[x + '.mean'].tolist()
	x_margin = ((data[x + '.r'] - data[x + '.l'])/3).tolist()

	y_mean = data[y + '.mean'].tolist()
	y_margin = ((data[y + '.r'] - data[y + '.l'])/3).tolist()

	colors = { 'Be Stars': 'b', 'Cepheid': 'g', 'Eclipsing Binaries': 'r', 'Long Periodic Variable': 'c', 'MicroLensing': 'm', 'None Variable': 'y', 'Quasar': 'k', 'RR Lyrae':'w' }
	class_list = data['class'].tolist()

	ells = []
	for i in xrange(len(x_mean)):
		ells.append( Ellipse(xy=[x_mean[i], y_mean[i]], width=x_margin[i], height=y_margin[i], angle=0, facecolor=colors[class_list[i]]) )

	fig = plt.figure(1)
	ax = fig.add_subplot(111)

	# plt.xlim(-0.1,0.8)
	# plt.ylim(-0.1,1.5)
	plt.xlim(min(x_mean) - max(x_margin), max(x_mean) + max(x_margin))
	plt.ylim(min(y_mean) - max(y_margin), max(y_mean) + max(y_margin))

	plt.xlabel(x)
	plt.ylabel(y, rotation = 'horizontal')

	plt.axhline(y = y_value, color = 'r')
	plt.axvline(x = x_value, color = 'r')

	for e in ells:
	    ax.add_artist(e)
	    e.set_clip_box(ax.bbox)
	    e.set_alpha(0.7)

	plt.show()
	# plt.savefig('utree ' + x + ' v/s ' + y + '.png')
	plt.close()

def graf_points( path, clases, x, y, x_value, y_value ):
	
	data = pd.read_csv(path)
	
	# Elijo solo algunas clases
	criterion = data['class'].map(lambda x: x in clases)
	data = data[criterion]

	colors = { 'Be Stars': 'b', 'Cepheid': 'g', 'Eclipsing Binaries': 'r', 'Long Periodic Variable': 'c', 'MicroLensing': 'm', 'None Variable': 'y', 'Quasar': 'k', 'RR Lyrae':'w' }

	fig = plt.figure(1)
	ax = fig.add_subplot(111)

	for c in clases:

		# Valores a graficar
		x_mean = data[data['class'] == c][x + '.mean'].tolist()
		y_mean = data[data['class'] == c][y + '.mean'].tolist()

		ax.scatter( x_mean, y_mean, c = colors[c] )	

	# plt.xlim(-0.1,0.8)
	# plt.ylim(-0.1,1.5)
	range_x = max(x_mean) - min(x_mean) / 100
	range_y = max(y_mean) - min(y_mean) / 100

	plt.xlim(min(x_mean) - range_x, max(x_mean) + range_x)
	plt.ylim(min(y_mean) - range_y, max(y_mean) + range_y)

	plt.xlabel(x)
	plt.ylabel(y, rotation = 'horizontal')

	plt.axhline(y = y_value, color = 'r')
	plt.axvline(x = x_value, color = 'r')

	plt.show()
	# plt.savefig('tree ' + x + ' v/s ' + y + '.png')
	plt.close()



if __name__ == '__main__':
	path = 'sets/macho random sampled 10 percent.csv'
	clases = [ 'Long Periodic Variable', 'MicroLensing', 'None Variable', 'RR Lyrae']

	x = 'CAR_mean_R'
	y = 'N_below_4_R'
	z = 'Period_SNR_R'

	###### Cortes arbol clasico ######

	x_value = -7.46
	y_value = 0.00
	z_value = 0.07

	graf_points(path, clases, x, y, x_value, y_value)
	graf_points(path, clases, x, z, x_value, z_value)
	graf_points(path, clases, y, z, y_value, z_value)

	###### Cortes arbol incierto ######

	x_value = -7.26
	y_value = 0.09
	z_value = 0.12

	graf_ellipses(path, clases, x, y, x_value, y_value)
	graf_ellipses(path, clases, x, z, x_value, z_value)
	graf_ellipses(path, clases, y, z, y_value, z_value)

	