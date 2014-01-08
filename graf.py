import pickle
import matplotlib.pyplot as plt
import tree

# RESULT_DIR = 'resultados var_comp/'
RESULT_DIR = 'resultados trust/'

# Cargo el arbol de desicion
# lector = open('arbol 80.pkl', 'r')
# clf = pickle.load(lector)
# lector.close()

clf = tree.Tree('confianza')

p = 100

# Cargo los resultados de la prediccion
lector = open( RESULT_DIR + 'result ' + str(p) +'.pkl', 'r')
result = pickle.load(lector)
lector.close()


# Para cada clase
for clase in range(2,10):

	valores_accuracy = []

	valores_recall = []

	x_values = []

	# Para cada porcentaje de confianza
	for i in range(100):

		# Obtengo las predicciones con una confianza mayor a cierto umbral
		porcentaje = float(i)/100
		aux = result[result['trust'] > porcentaje]

		matrix = clf.confusion_matrix(aux)

		# Si la precision es menor que cero, es porque no habian datos que superaran tal nivel de confianza
		precision = clf.accuracy(matrix, clase)
		if precision >= 0:
			valores_accuracy.append(precision)

			valores_recall.append(clf.recall(matrix, clase))

			x_values.append(porcentaje)


	# Grafico los valores obtenidos
	plt.figure(clase)
	plt.plot( x_values, valores_accuracy, 'bo')

	plt.ylim(0.0, 1.0)
	plt.xlim(0.0, 1.0)

	plt.title( 'Class ' + str(clase) + ' accuracy v/s curve percentage')
	plt.xlabel( 'Lightcurve percentage')
	plt.ylabel( 'Accuracy' )

	plt.savefig('Clase ' + str(clase) + ' accuracy ' + str(p) + '%.png')
	plt.close()


	plt.figure(clase)
	plt.plot( x_values, valores_recall, 'bo')

	plt.ylim(0.0, 1.0)
	plt.xlim(0.0, 1.0)

	plt.title( 'Class ' + str(clase) + ' recall v/s curve percentage')
	plt.xlabel( 'Lightcurve percentage')
	plt.ylabel( 'Recall' )

	plt.savefig('Clase ' + str(clase) + ' recall ' + str(p) + '%.png')
	plt.close()
	
