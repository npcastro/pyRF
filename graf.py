# Guardo los metodos basicos para graficar y mantener el codigo dry

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

if __name__ == '__main__':
	path = 'sets/artificial random 10.csv'
	data = pd.read_csv(path)

	colors = {'blue': 'b', 'red': 'r'}

	fig = plt.figure(1)
	ax = fig.add_subplot(111)

	draw_points(ax, data, 'Feature 1', 'Feature 2', colors)

	plt.show()

