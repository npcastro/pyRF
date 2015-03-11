# Este script lo ocupe para generar graficos comparativos entre los resultados obtenidos 
# entre el arbol normal y el arbol con incertidumbre

import pandas as pd
import tree
import matplotlib.pyplot as plt

def get_avance(directory):

	result = pd.read_csv(directory + 'result 5.pkl', index_col=0)
	clases = result.original.unique().tolist()

	# Hago un diccionario con listas para cada clase. Ahi guardo los indicadores a 
	# medida que aumenta la incertidumbre

	avances = {c: [] for c in clases}

	niveles = range(5,70,5)

	for i in niveles:

		result = pd.read_csv(directory + 'result ' + str(i) +'.pkl', index_col=0)

		clf = tree.Tree('uncertainty', max_depth=10,
		                        min_samples_split=20, most_mass_threshold=0.9, min_mass_threshold=0.10,
		                        min_weight_threshold=0.01)

		matrix = clf.hard_matrix(result)

		# p = {c: clf.precision(matrix, c) for c in clases}
		# r = {c: clf.recall(matrix, c) for c in clases}
		f = {c: clf.f_score(matrix, c) for c in clases}

		for c in clases:
			avances[c].append(f[c])

	return avances

# Voy a un directorio con resultados para distintos niveles de completitud
# directorio_u = 'Resultados/Comparacion/iris/Fijo/U/'
# directorio_normal = 'Resultados/Comparacion/iris/Fijo/Normal/'

# directorio_u = 'Resultados/Comparacion/iris/Random/U/'
# directorio_normal = 'Resultados/Comparacion/iris/Random/Normal/'

directorio_u = 'Resultados/Comparacion/macho/Random/U/'
directorio_normal = 'Resultados/Comparacion/macho/Random/Normal/'

avance_u = get_avance(directorio_u)
avance_normal = get_avance(directorio_normal)

# for c in avance_u.keys():

# 	plt.figure()
# 	plt.plot( range(5,70,5), avance_u[c], 'bo-', label='UTree')
# 	plt.plot( range(5,70,5), avance_normal[c], 'ro-', label='Classic')

# 	plt.ylim(0.0,1.0)

# 	plt.title( str(c) + ' F-Score v/s Data uncertainty')
# 	plt.xlabel( 'Uncertainty used')
# 	plt.ylabel( 'F-Score' )
# 	plt.legend()

# 	# plt.savefig('Resultados/Comparacion/Graficos/iris/Fijo/' + str(c) + ' fscore.png')
# 	# plt.savefig('Resultados/Comparacion/Graficos/iris/Random/' + str(c) + ' fscore.png')
# 	plt.savefig('Resultados/Comparacion/Graficos/macho/Random/' + str(c) + ' fscore.png')
# 	# plt.show()
# 	plt.close()



