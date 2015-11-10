# Filtro el set original de datos para la metaclasificacion

import pandas as pd

set_path = '/n/home09/ncastro/workspace/Features/sets/'
data = pd.read_csv(set_path)

data = data.dropna(axis=0, how='any')

facil = ['non_variables', 'CEPH', 'quasar_lc', 'longperiod_lc', 'RRL', 'microlensing_lc']
dificil = ['Be_lc','EB']

# Los datos dificiles de clasificar
data_b = data[data['class'].apply(lambda x: True if x in dificil else False)]

# Los datos faciles de clasificar
data_c = data[data['class'].apply(lambda x: True if x not in dificil else False)]

# El set con solo dos clases
y = data['class']
y_a = y.apply(lambda x: 'facil' if x in facil else 'dificil')
data['class'] = y_a

# Los guardo con los indices por si los necesito despues para reconocer las tuplas en la 
# Metaclasificacion
data.to_csv('/n/home09/ncastro/workspace/Features/sets/Macho_a.csv')
data_b.to_csv('/n/home09/ncastro/workspace/Features/sets/Macho_b.csv')
data_c.to_csv('/n/home09/ncastro/workspace/Features/sets/Macho_c.csv')