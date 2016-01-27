import pickle
import matplotlib.pyplot as plt
import metrics

def graf(clase, x_values, y_values, y_label):

	# Grafico los valores obtenidos
	# plt.figure(clase)
	plt.plot( x_values, y_values, 'o', label=clase)

	plt.ylim(0.5, 1.0)
	plt.xlim(0.0, 1.0)

	# plt.title( y_label +  ' v/s Prediction Certainty')
	plt.xlabel( 'Minimum Probability Considered')
	plt.ylabel( y_label )

	plt.legend(loc='lower left', prop={'size':10})

	# plt.savefig('Resultados/macho ' + str(clase) + ' ' + y_label + ' .png')
	# plt.show()
	# plt.close()

if __name__ == '__main__':

	RESULT_DIR = 'output/macho/'

	lector = open( RESULT_DIR + 'result random.pkl', 'r')
	result = pickle.load(lector)
	lector.close()

	# Para cada clase

	a = plt.figure()

	for clase in result['original'].unique().tolist():

		valores_accuracy = []
		valores_recall = []
		valores_fscore = []
		x_values = []
		x_values_fscore = []

		# Para cada porcentaje de confianza
		for i in xrange(100):

			# Obtengo las predicciones con una confianza mayor a cierto umbral
			porcentaje = float(i)/100

			aux = result[result['trust'] > porcentaje]

			# matrix = metrics.confusion_matrix(aux)
			matrix = metrics.hard_matrix(aux)

			# Si la precision es menor que cero, es porque no habian datos que superaran tal nivel de confianza
			precision = metrics.accuracy(matrix, clase)
			if precision >= 0:
				valores_accuracy.append(precision)
				valores_recall.append(metrics.recall(matrix, clase))
				x_values.append(porcentaje)

			# Si el f_score es menor que cero, es porque no habian datos que superaran tal nivel de confianza
			f_score = metrics.f_score(matrix, clase)
			if f_score >= 0:
				valores_fscore.append(f_score)
				x_values_fscore.append(porcentaje)			

		#graf(clase, x_values, valores_accuracy, 'Accuracy')
		graf(clase, x_values, valores_recall, 'Recall')
		#graf(clase, x_values_fscore, valores_fscore, 'F-Score')
		print 'a'

	plt.show()