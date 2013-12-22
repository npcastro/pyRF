import pandas as pd
from tree import *

if __name__ == '__main__':

	# nombres = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'class']
	# data = pd.read_csv('abalone.data', header = None, names = nombres)

	#nombres = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
	nombres = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class', 'sepal length_conf', 'sepal width_conf', 'petal length_conf', 'petal width_conf']
	#data = pd.read_csv('iris.data', header=None, names=nombres)
	data = pd.read_csv('iris_comp.data', header=None, names=nombres)

	#clf = Tree('gain')
	clf = Tree('confianza')
	clf.fit(data)