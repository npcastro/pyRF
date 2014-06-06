import pickle
import matplotlib.pyplot as plt
import tree

@profile
def graf(clase, x_values, y_values, y_label):

	# Grafico los valores obtenidos
	plt.figure(clase)
	plt.plot( x_values, y_values, 'bo')

	plt.ylim(0.0, 1.0)
	plt.xlim(0.0, 1.0)

	plt.title( 'Class ' + str(clase) + ' ' + y_label +  ' v/s Prediction Certainty')
	plt.xlabel( 'Minimum Probability Considered')
	plt.ylabel( y_label )

	plt.savefig('Resultados/macho ' + str(clase) + ' ' + y_label + ' .png')
	# plt.show()
	plt.close()

if __name__ == '__main__':

	RESULT_DIR = 'output/macho/'
		
	# Inicializo un arbol cualquiera para tener sus metodos
	clf = tree.Tree('uncertainty')

	lector = open( RESULT_DIR + 'result random.pkl', 'r')
	result = pickle.load(lector)
	lector.close()

	# Para cada clase

	for clase in set(result['original'].values.tolist()):

		valores_accuracy = []
		valores_recall = []
		valores_fscore = []
		x_values = []
		x_values_fscore = []

		# Para cada porcentaje de confianza
		for i in range(100):

			# Obtengo las predicciones con una confianza mayor a cierto umbral
			porcentaje = float(i)/100
			aux = result[result['trust'] > porcentaje]

			# matrix = clf.confusion_matrix(aux)
			matrix = clf.hard_matrix(aux)

			# Si la precision es menor que cero, es porque no habian datos que superaran tal nivel de confianza
			precision = clf.accuracy(matrix, clase)
			if precision >= 0:
				valores_accuracy.append(precision)
				valores_recall.append(clf.recall(matrix, clase))
				x_values.append(porcentaje)

			# Si el f_score es menor que cero, es porque no habian datos que superaran tal nivel de confianza
			f_score = clf.f_score(matrix, clase)
			if f_score >= 0:
				valores_fscore.append(f_score)
				x_values_fscore.append(porcentaje)			

		graf(clase, x_values, valores_accuracy, 'Accuracy')
		graf(clase, x_values, valores_recall, 'Recall')
		graf(clase, x_values_fscore, valores_fscore, 'F-Score')