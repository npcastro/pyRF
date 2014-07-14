# Guardo los metodos basicos para graficar y mantener el codigo dry

import matplotlib.pyplot as plt

def draw_points( x, y, x_value, y_value ):

	# plt.xlim(min(x_mean) - max(x_margin), max(x_mean) + max(x_margin))
	# plt.ylim(min(y_mean) - max(y_margin), max(y_mean) + max(y_margin))

	# fig = plt.figure(1)
	# ax = fig.add_subplot(111)

	# # Graficar
	# for c in clases:

	# 	x_mean = data[data['class'] == c][x + '.mean'].tolist()
	# 	x_margin = ((data[x + '.r'] - data[x + '.l'])/1).tolist()
	# 	y_mean = data[data['class'] == c][y + '.mean'].tolist()
	# 	y_margin = ((data[y + '.r'] - data[y + '.l'])/1).tolist()

	# 	ax.scatter( x_mean, y_mean, c = colors[c] )	

	# plt.xlabel(x)
	# plt.ylabel(y, rotation = 'horizontal')

	# plt.axhline(y = y_value, color = 'r')
	# plt.axvline(x = x_value, color = 'r')

	# plt.title('tree ' + x + ' vs ' + y + '.png')
	# # plt.show()
	# plt.savefig('Resultados/tree ' + x + ' vs ' + y + '.png')
	# plt.close()