import pandas as pd
import tree

def fix_format(directory):

	result = pd.read_csv(directory + 'result 5.pkl', index_col=0)
	clases = result.original.unique().tolist()

	p_dict = {c: [] for c in clases}
	r_dict = {c: [] for c in clases}
	f_dict = {c: [] for c in clases}

	niveles = range(5,70,5)

	for i in niveles:

		result = pd.read_csv(directory + 'result ' + str(i) +'.pkl', index_col=0)

		clf = tree.Tree('uncertainty', max_depth=10,
		                        min_samples_split=20, most_mass_threshold=0.9, min_mass_threshold=0.10,
		                        min_weight_threshold=0.01)

		matrix = clf.hard_matrix(result)

		p = {c: clf.precision(matrix, c) for c in clases}
		r = {c: clf.recall(matrix, c) for c in clases}
		f = {c: clf.f_score(matrix, c) for c in clases}

		for c in clases:
			p_dict[c].append(p[c])
			r_dict[c].append(r[c])
			f_dict[c].append(f[c])

	return p_dict, r_dict, f_dict

niveles = range(5,70,5)

# directorio = 'Resultados/Comparacion/iris/Fijo/Normal/'
# directorio = 'Resultados/Comparacion/iris/Random/U/'
# directorio = 'Resultados/Comparacion/iris/Random/Normal/'

# directorio = 'Resultados/Comparacion/macho/Random/U/'
directorio = 'Resultados/Comparacion/macho/Random/Normal/'

p, r, f = fix_format(directorio)

p_df = pd.DataFrame(p, index = niveles)
p_df.to_csv(directorio + 'precision.csv')
r_df = pd.DataFrame(r, index = niveles)
r_df.to_csv(directorio + 'recall.csv')
f_df = pd.DataFrame(f, index = niveles)
f_df.to_csv(directorio + 'f_score.csv')


