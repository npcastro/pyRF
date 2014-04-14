import tree
import pandas as pd
import numpy as np
import pickle
from sklearn import cross_validation


folds = 4

porcentajes = [5, 10, 15, 20, 25, 30, 35,40,45,50]

for p in porcentajes:

	path = "sets/macho " + str(p) + ".csv"


	data = pd.read_csv(path)
	data['weight'] = data['weight'].astype(float)

	skf = cross_validation.StratifiedKFold(data['class'], n_folds=folds)

	results = []
	count = 1
	for train_index, test_index in skf:
		print 'Fold: ' + str(count)
		count += 1
		train, test = data.iloc[train_index], data.iloc[test_index]

		clf = None
		# clf = tree.Tree('confianza')
		clf = tree.Tree('gain')
		# clf = tree.Tree('uncertainty')

		clf.fit(train)
		results.append(clf.predict_table(test))

	result = pd.concat(results)
	matrix = clf.confusion_matrix(result)	
	hard = clf.hard_matrix(result)

	output = open( 'output/macho_result '+ str(p) + '.pkl', 'w')
	pickle.dump(result, output)
	output.close()

	output = open( 'output/arbol_macho ' + str(p) + '.pkl', 'w')
	pickle.dump(clf, output)
	output.close()

	result.to_csv('output/macho_result ' + str(p) + '.csv', index=False)