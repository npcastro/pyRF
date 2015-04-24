# 18 de marzo 
# Este script lo ocupo para generar graficos comparativos de los resultados obtenidos 
# entre el arbol normal y el arbol con incertidumbre

import pandas as pd
import tree
import matplotlib.pyplot as plt


# Voy a un directorio con resultados para distintos niveles de completitud

directorio_u = 'Resultados/Comparacion/Macho/Random II/U/'
directorio_normal = 'Resultados/Comparacion/Macho/Random II/Normal/'

avance_u_soft = pd.read_csv(directorio_u + 'f_score.csv', index_col=0).to_dict(orient='list')
avance_normal = pd.read_csv(directorio_normal + 'f_score.csv', index_col=0).to_dict(orient='list')


# result_dir = 'Resultados/Comparacion/Graficos/hard vs normal/Macho/Random II/'
# result_dir = 'Resultados/Comparacion/Graficos/soft vs normal/Macho/Random II/'
result_dir = 'Resultados/Comparacion/Graficos/todos/Macho/Random II/'

for c in avance_u_soft.keys():
# for c in avance_normal.keys():

	plt.figure()
	# plt.plot( range(5,70,5), avance_u_hard[c], 'bo-', label='UTree hard')

	plt.plot( range(5,70,5), avance_u_soft[c], 'go-', label='UTree soft')
	plt.plot( range(5,70,5), avance_normal[c], 'ro-', label='Classic')

	plt.ylim(0.0,1.0)

	plt.title( str(c) + ' F-Score v/s Data uncertainty')
	plt.xlabel( 'Uncertainty used')
	plt.ylabel( 'F-Score' )
	plt.legend(loc=3)

	plt.savefig(result_dir + str(c) + ' fscore.png')

	# plt.show()
	plt.close()



