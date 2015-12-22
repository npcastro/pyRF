# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
import tree
import metrics

# result_dir = '/Users/npcastro/Dropbox/Resultados/MACHO/UTree/GP/Predicciones/'
# save_dir = '/Users/npcastro/Dropbox/Resultados/MACHO/Graficos/'

result_dir = '/Users/npcastro/Dropbox/Resultados/EROS/UTree/GP/Predicciones/'
save_dir = '/Users/npcastro/Dropbox/Resultados/EROS/Graficos/'

# No esta terminado pero tiene que haber una manera de hacer el proceso mas rápido
def find_indexes(lista):
    indexes = []

    limites = [x/100.0 for x in range(1,101)]

    aux = 0

    for i, x in enumerate(lista):
        if x > limites[aux]:
            aux += 1
            indexes.append(i)
    return indexes

# regular_fscore = pd.read_csv('/Users/npcastro/Dropbox/Resultados/MACHO/Comparacion/Tree/Metricas/f_score.csv', index_col=0)
# regular_fscore = pd.read_csv('/Users/npcastro/Dropbox/Resultados/MACHO/Tree/Regular/Metricas/f_score.csv', index_col=0)
regular_fscore = pd.read_csv('/Users/npcastro/Dropbox/Resultados/EROS/Tree/Regular/Metricas/f_score.csv', index_col=0)

for p in xrange(25, 105, 5):

    if p == 50:
        continue

    print str(p) + '%'
    
    result = pd.read_csv(result_dir + 'result_' + str(p) +'.csv', index_col=0)

    clases = result['original'].unique().tolist()

    x_values = {clase: [] for clase in clases}
    valores_fscore = {clase: [] for clase in clases}

    result = result.sort('trust', axis=0)

    for i in xrange(100):
        
        # Obtengo las predicciones con una confianza mayor a cierto umbral
        trust_threshold = float(i)/100
        result = result[result['trust'] > trust_threshold]

        matrix = metrics.hard_matrix(result)

        # Si el f_score es menor que cero, es porque no habian datos que superaran tal nivel de confianza
        f_scores = {clase: metrics.f_score(matrix, clase) for clase in clases}

        for clase in clases:
            if f_scores[clase] >= 0:
                valores_fscore[clase].append(f_scores[clase])
                x_values[clase].append(trust_threshold)

    for clase in clases:
        x_list = x_values[clase]
        y_list = valores_fscore[clase]
        
        plt.figure(clase)
        plt.plot( x_list, y_list, '-ob')

        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.axhline(regular_fscore.loc[p][clase], ls='--', color='r')

        plt.title( 'Class ' + str(clase) + ' F-Score v/s Prediction Certainty')
        plt.xlabel( 'Minimum Probability Considered')
        plt.ylabel( 'F-Score' )

        plt.savefig(save_dir + str(p) + '%/' + str(clase) + ' fscore progress.png')
        plt.close()