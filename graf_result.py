import pandas as pd
import matplotlib.pyplot as plt


regular_progress = pd.read_csv('Resultados/Regular/Metricas/f_score.csv', index_col=0)
u_progress = pd.read_csv('Resultados/GP/Metricas/f_score.csv', index_col=0)


for clase in regular_progress.columns.tolist():

	x_values = range(5, 105, 5)
	regular_values = regular_progress[clase].tolist()
	u_values = u_progress[clase].tolist()

	plt.figure(clase)
	plt.ylim(0.0, 1.0)
	plt.title( 'Class ' + str(clase) + ' F-Score v/s Lightcurve percentage')
	plt.xlabel( 'Percentage of observations')
	plt.ylabel( 'F-Score' )

	
	plt.plot( x_values, regular_values, '-o')
	plt.plot( x_values, u_values, '-or' )

	# plt.savefig('Resultados/Regular/Graficos/Clase ' + str(clase) + ' fscore.png')
	# plt.savefig('Resultados/GP/Graficos/Clase ' + str(clase) + ' fscore.png')
	plt.savefig('Resultados/Graficos/Clase ' + str(clase) + ' fscore.png')

	plt.close()