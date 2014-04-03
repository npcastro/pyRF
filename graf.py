import pickle
import matplotlib.pyplot as plt
import tree

RESULT_DIR = 'resultados var_comp/'
# RESULT_DIR = 'resultados trust/'
# RESULT_DIR = 'resultados new_var/'

# Porcentaje del arbol a ocupar
p = 20

# Inicializo un arbol cualquiera para tener sus metodos
clf = tree.Tree('confianza')

# Cargo el arbol de desicion
# lector = open('arbol 80.pkl', 'r')
# clf = pickle.load(lector)
# lector.close()


# Cargo los resultados de la prediccion
lector = open( RESULT_DIR + 'result ' + str(p) +'.pkl', 'r')
# lector = open( 'Result especial.pkl', 'r')
result = pickle.load(lector)
lector.close()


# Para cada clase
for clase in range(2,10):

	# valores_accuracy = []
	# valores_recall = []

	valores_fscore = []

	x_values = []

	# Para cada porcentaje de confianza
	for i in range(100):

		# Obtengo las predicciones con una confianza mayor a cierto umbral
		porcentaje = float(i)/100
		aux = result[result['trust'] > porcentaje]

		# porcentaje = 1.0 - float(i)/100
		# aux = result[result['trust'] < porcentaje]

		matrix = clf.confusion_matrix(aux)

		# Si la precision es menor que cero, es porque no habian datos que superaran tal nivel de confianza
		# precision = clf.accuracy(matrix, clase)
		# if precision >= 0:
		# 	valores_accuracy.append(precision)
		# 	valores_recall.append(clf.recall(matrix, clase))
		# 	x_values.append(porcentaje)

		# Si el f_score es menor que cero, es porque no habian datos que superaran tal nivel de confianza
		f_score = clf.f_score(matrix, clase)
		if f_score >= 0:
			valores_fscore.append(f_score)
			x_values.append(porcentaje)			


	# Grafico los valores obtenidos
	plt.figure(clase)
	plt.plot( x_values, valores_fscore, 'bo')

	plt.ylim(0.0, 1.0)
	plt.xlim(0.0, 1.0)

	plt.title( 'Class ' + str(clase) + ' F-Score v/s Prediction Certainty')
	plt.xlabel( 'Minimum Stability considered')
	plt.ylabel( 'F-Score' )

	plt.savefig('Clase ' + str(clase) + ' fscore ' + str(p) + '%.png')
	plt.close()

if __name__ == '__main__':
	pass