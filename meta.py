# coding=utf-8
# Metaclasificador

# Esta compuesto por tres clasificadores. 
# El primero se entrena para separar los datos entre dos grupos de clases que son pasadas como
# parámetro
# El segundo y el tercero se entrenan para clasificar solo entre los subgrupos que se forman
# despues del filtro que hace el primer clasificador

# La predicción es siempre probabilistica


class MetaClassifier:

	def __init__(self, clf_filter, clf_left, clf_right):
		self.clf_filter = clf_filter
		self.clf_left = clf_left
		self.clf_right = clf_right

	def fit(self, data, y):
		pass

	def predict(self, data, y):
		pass