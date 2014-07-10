# Toma un path a un dataset normal y le hace modificiaciones artificiales

import pandas as pd
import tree
import random

path = 'sets/macho random sampled 10 folds.csv'
data = pd.read_csv(path)

# Elimino incertidumbre de algunas variables para algunas clases
data['N_below_4_B.std'][data['class'] != 'RR Lyrae'] = 0.0000001
data['Period_SNR_R.std'][data['class'] != 'RR Lyrae'] = 0.0000001

# Desplazo los puntos de la clase Long Periodic Variable
data['N_below_4_B.mean'][data['class'] == 'Long Periodic Variable'] =  data['N_below_4_B.mean'][data['class'] == 'Long Periodic Variable'] - 0.01
data['N_below_4_B.r'][data['class'] == 'Long Periodic Variable'] = data['N_below_4_B.r'][data['class'] == 'Long Periodic Variable'] - 0.01
data['N_below_4_B.l'][data['class'] == 'Long Periodic Variable'] = data['N_below_4_B.l'][data['class'] == 'Long Periodic Variable'] - 0.01

# Desorden aleatorio de la clase RR Lyrae
aux = data['N_below_4_B.mean'][data['class'] == 'RR Lyrae'].map(lambda x: float(random.randrange(-51, 51, 1)) / 1000)

data['N_below_4_B.mean'][data['class'] == 'RR Lyrae'] =  data['N_below_4_B.mean'][data['class'] == 'RR Lyrae'] + aux
data['N_below_4_B.r'][data['class'] == 'RR Lyrae'] = data['N_below_4_B.r'][data['class'] == 'RR Lyrae'] + aux
data['N_below_4_B.l'][data['class'] == 'RR Lyrae'] = data['N_below_4_B.l'][data['class'] == 'RR Lyrae'] + aux


# data.to_csv('sets/macho std variable.csv')
data.to_csv('sets/macho artificial final.csv', index=False)