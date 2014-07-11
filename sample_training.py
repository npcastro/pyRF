# Toma un path a un directorio con varios sets de entrenamiento
# y entrena un arbol con todos los sets que hay dentro.

# Hay que pasar solo dos posibles features, para que siempre se elijan 
# y asi poder evaluar como cambian los cortes que elige el arbol

import pandas as pd
import tree
import os

def get_cut(node):
	return (node.feat_name, float("%.2f" % clf.root.feat_value))


if __name__ == '__main__':
	directory = 'sets/sampling'

	x = 'N_below_4_B'
	y = 'Period_SNR_R'

	for root, dirs, files in os.walk(directory):
		training_sets_paths = [directory + '/' + f for f in files if not f[0] == '.']

	root_cuts = []
	right_cuts = []
	left_cuts = []

	for path in training_sets_paths:

		data = pd.read_csv(path)

		# Elijo solo algunas clases
		clases = [ 'Long Periodic Variable', 'None Variable', 'RR Lyrae']
		criterion = data['class'].map(lambda x: x in clases)
		data = data[criterion]

		# Dejo solo algunas features
		columns = ['N_below_4_B.mean', 'Period_SNR_R.mean', 'class']
		data = data[columns]

		# Entreno arbol
		clf = tree.Tree('gain', max_depth=3)
		clf.fit(data)

		root_cuts.append( get_cut(clf.root))
		left_cuts.append( get_cut(clf.root.left))
		right_cuts.append( get_cut(clf.root.right))

		
