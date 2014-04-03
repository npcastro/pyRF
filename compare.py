# Entreno un arbol basado solo en ganancias de informacion y pero despues predigo considerando la incerteza
# Sirve para poder comparar si hay una mejora en entrenar considerando las incertezas

import pickle
import matplotlib.pyplot as plt
import tree
import pandas as pd

# Cambia el criterio de todo un arbol para que utilice confianzas
def change(node):
	node.criterium = 'confianza'
	if node.is_leaf:
		pass
	else:
		change(node.right)
		change(node.left)

RESULT_DIR = 'resultados new_var/'

# Entreno un arbol con el 100% de las curvas
path_train = "/Users/npcastro/workspace/Features/Entrenamiento new_var/Entrenamiento " + str(100) + ".txt"
path_test = "/Users/npcastro/workspace/Features/Entrenamiento new_var/Entrenamiento " + str(40) + ".txt"

nombres = ['Macho_id', 'Sigma_B', 'Sigma_B_comp', 'Eta_B', 'Eta_B_comp', 'stetson_L_B', 'stetson_L_B_comp', 'CuSum_B', 'CuSum_B_comp', 'B-R', 'B-R_comp', 'stetson_J', 'stetson_J_comp', 'stetson_K', 'stetson_K_comp', 'skew', 'skew_comp', 'kurt', 'kurt_comp', 'std', 'std_comp', 'beyond1_std', 'beyond1_std_comp', 'max_slope', 'max_slope_comp', 'amplitude', 'amplitude_comp', 'med_abs_dev', 'med_abs_dev_comp', 'class']

data_train = pd.read_csv(path_train, sep=' ', header=None, names=nombres, skiprows=1, index_col=0)
data_test = pd.read_csv(path_test, sep=' ', header=None, names=nombres, skiprows=1, index_col=0)

train = pd.DataFrame()
test = pd.DataFrame()

# Genero un set de test con el 10% de los datos de cada clase

for i in data_train['class'].unique():

    aux_train = data_train[data_train['class'] == i]
    aux_test = data_test[data_test['class'] == i]

    total = len(aux_train.index)
    fraccion = total / 10

    train = train.append(aux_train.iloc[0:-fraccion])
    test = test.append(aux_test.iloc[-fraccion:])

clf = tree.Tree('gain')
clf.fit(train)


# Cambio el tipo del arbol para que prediga considerando las confianzas
clf.criterium = 'confianza'
change(clf.root)

# Predigo
result = clf.predict_table(test)
matrix = clf.confusion_matrix(result)

# Serializo los resultados con pickle
output = open( 'Arbol 100 gain.pkl', 'w')
pickle.dump(clf, output)
output.close()

output = open( 'Result especial.pkl', 'w')
pickle.dump(result, output)
output.close()


