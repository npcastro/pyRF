# Toma un path a un dataset normal y le hace modificiaciones artificiales

import pandas as pd
import tree
import random

import matplotlib.pyplot as plt

import graf

import numpy as np

# path = 'sets/macho random sampled 10 folds.csv'
# data = pd.read_csv(path)

# # Elimino incertidumbre de algunas variables para algunas clases
# data['N_below_4_B.std'][data['class'] != 'RR Lyrae'] = 0.0000001
# data['Period_SNR_R.std'][data['class'] != 'RR Lyrae'] = 0.0000001

# # Desplazo los puntos de la clase Long Periodic Variable
# data['N_below_4_B.mean'][data['class'] == 'Long Periodic Variable'] =  data['N_below_4_B.mean'][data['class'] == 'Long Periodic Variable'] - 0.01
# data['N_below_4_B.r'][data['class'] == 'Long Periodic Variable'] = data['N_below_4_B.r'][data['class'] == 'Long Periodic Variable'] - 0.01
# data['N_below_4_B.l'][data['class'] == 'Long Periodic Variable'] = data['N_below_4_B.l'][data['class'] == 'Long Periodic Variable'] - 0.01

# # Desorden aleatorio de la clase RR Lyrae
# aux = data['N_below_4_B.mean'][data['class'] == 'RR Lyrae'].map(lambda x: float(random.randrange(-51, 51, 1)) / 1000)

# data['N_below_4_B.mean'][data['class'] == 'RR Lyrae'] =  data['N_below_4_B.mean'][data['class'] == 'RR Lyrae'] + aux
# data['N_below_4_B.r'][data['class'] == 'RR Lyrae'] = data['N_below_4_B.r'][data['class'] == 'RR Lyrae'] + aux
# data['N_below_4_B.l'][data['class'] == 'RR Lyrae'] = data['N_below_4_B.l'][data['class'] == 'RR Lyrae'] + aux


# # data.to_csv('sets/macho std variable.csv')
# data.to_csv('sets/macho artificial final.csv', index=False)


def random_circle_point(x, y, radius):
	"""Genera un punto aleatorio dentro de un circulo centrado en x,y
	"""
	a = 2 * np.pi * random.random()
	r = np.sqrt(random.random())
	x = (radius * r ) * np.cos(a) + x + random.uniform(-radius/15, radius/15)
	y = (radius * r ) * np.sin(a) + y + +random.uniform(- radius/6, radius/6)

	return x, y

if __name__ == '__main__':
	
	# Diccionario donde almacenar los valores
	data = { 'Feature 1': [], 'Feature 2': [], 'class': []}

	n_samples_per_class = 500

	for i in xrange(n_samples_per_class):
		x, y = random_circle_point(2,2.5, 1)
		data['Feature 1'].append(x)
		data['Feature 2'].append(y)
		data['class'].append('blue')

	for i in xrange(n_samples_per_class):
		x, y = random_circle_point(4,4.5, 1)
		data['Feature 1'].append(x)
		data['Feature 2'].append(y)
		data['class'].append('red')

	data = pd.DataFrame(data)

	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	colors = {'blue': 'b', 'red': 'r'}

	graf.draw_points(ax, data, 'Feature 1', 'Feature 2', colors)
	graf.add_names('F1 vs F2', 'Feature 1', 'Feature 2')

	plt.xlim(0, 6)
	plt.ylim(0, 6)
	# plt.show()

	data.to_csv('sets/artificial.csv', index=False)

