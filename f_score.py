# Toma un directorio de resultados y arma un csv con el progreso
# de los f_score

import metrics

import pandas as pd

# path = '/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Sampled/uniform/UF/'
# path = '/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Comparacion/Tree/'
# path = '/n/seasfs03/IACS/TSC/ncastro/Resultados/EROS/Tree/Completed/'
# path = '/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Tree/Completed/'
# path = '/n/seasfs03/IACS/TSC/ncastro/Resultados/EROS/Tree/Regular/'
# path = '/Users/npcastro/Dropbox/Resultados/EROS/Sampled/uniform/UF/'
# path = '/Users/npcastro/Desktop/UF/'
path = '/n/seasfs03/IACS/TSC/ncastro/Resultados/MACHO/Sampled/uniform/Montecarlo/'

result_dir = path + 'Predicciones/'

p_dict = {}
r_dict = {}
f_dict = {}
w_dict = {}

for percentage in xrange(5, 10, 5):

    result = pd.read_csv(result_dir + 'result_' + str(percentage) + '.csv', index_col=0)
    matrix = metrics.confusion_matrix(result)
    matrix.to_csv(path + 'Metricas/confusion_' + str(percentage) + '.csv')
    # matrix = metrics.hard_matrix(result)

    w_dict[percentage] = metrics.weighted_f_score(matrix)

    clases = matrix.columns.tolist()
    p = [metrics.precision(matrix, c) for c in clases]
    r = [metrics.recall(matrix, c) for c in clases]
    f = [metrics.f_score(matrix, c) for c in clases]

    p_dict[percentage] = p
    r_dict[percentage] = r
    f_dict[percentage] = f

save_dir = path + 'Metricas/'

w_df = pd.DataFrame.from_dict(w_dict, orient='index')
w_df.columns = ['f_score']
w_df = w_df.sort_index(ascending=True)
w_df = w_df.fillna(value=0.0)
w_df.to_csv(save_dir + 'weight_fscore.csv')

p_df = pd.DataFrame.from_dict(p_dict, orient='index')
p_df.columns = clases
p_df = p_df.sort_index(ascending=True)
p_df = p_df.fillna(value=0.0)
p_df.to_csv(save_dir + 'precision.csv')

r_df = pd.DataFrame.from_dict(r_dict, orient='index')
r_df.columns = clases
r_df = r_df.sort_index(ascending=True)
r_df = r_df.fillna(value=0.0)
r_df.to_csv(save_dir + 'recall.csv')

f_df = pd.DataFrame.from_dict(f_dict, orient='index')
f_df.columns = clases
f_df = f_df.sort_index(ascending=True)
f_df = f_df.fillna(value=0.0)
f_df.to_csv(save_dir + 'f_score.csv')

