# Toma un directorio de resultados y arma un csv con el progreso
# de los f_score

import tree

import pandas as pd

clf = tree.Tree('gain', max_depth=10, min_samples_split=20)

result_dir = 'Resultados/'

for p in xrange(5,105,5):



	matrix = clf.confusion_matrix(result)

	clases = matrix.columns.tolist()
	p = [clf.precision(matrix, c) for c in clases]
	r = [clf.recall(matrix, c) for c in clases]
	f = [clf.f_score(matrix, c) for c in clases]