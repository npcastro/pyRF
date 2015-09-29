# Toma un directorio de resultados y arma un csv con el progreso
# de los f_score

import tree

import pandas as pd

clf = tree.Tree('gain', max_depth=10, min_samples_split=20)

# result_dir = 'Resultados/Regular/Predicciones/'
result_dir = 'Resultados/GP/Predicciones/'

p_dict = {}
r_dict = {}
f_dict = {}

for percentage in xrange(5,105,5):

	result = pd.read_csv(result_dir + 'result_' + str(percentage) + '.csv', index_col=0)

	matrix = clf.confusion_matrix(result)

	clases = matrix.columns.tolist()
	p = [clf.precision(matrix, c) for c in clases]
	r = [clf.recall(matrix, c) for c in clases]
	f = [clf.f_score(matrix, c) for c in clases]

	p_dict[percentage] = p
	r_dict[percentage] = r
	f_dict[percentage] = f

# save_dir = 'Resultados/Regular/Metricas/'
save_dir = 'Resultados/GP/Metricas/'

p_df = pd.DataFrame.from_dict(p_dict, orient='index')
p_df.columns = clases
p_df = p_df.sort_index(ascending=True)
p_df = p_df.fillna(value=0.0)
p_df.to_csv(save_dir + 'precision.csv')

r_df = pd.DataFrame.from_dict(r_dict, orient='index')
r_df.columns = clases
r_df = r_df.sort_index(ascending=True)
r_df = p_df.fillna(value=0.0)
r_df.to_csv(save_dir + 'recall.csv')

f_df = pd.DataFrame.from_dict(f_dict, orient='index')
f_df.columns = clases
f_df = f_df.sort_index(ascending=True)
f_df = p_df.fillna(value=0.0)
f_df.to_csv(save_dir + 'f_score.csv')

