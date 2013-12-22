from __future__ import division
import numpy as np
from scipy import stats
# data es un dataframe que tiene que contener una columna class. La cual el arbol intenta predecir.
# podria pensar en relajar esto y simplemente indicar cual es la variable a predecir.

class Node:

	def __init__(self, data, criterium):

		self.data = data

		self.is_leaf = False
		self.clase = ''
		self.feat_name = ""
		self.feat_value = None
		self.left = None
		self.right = None
		self.criterium = criterium		# 'gain'
		self.entropia = self.entropy(data)

		# Si es necesario particionar el nodo, llamo a split para hacerlo
		if self.check_data():
			self.split()
			menores = self.data[self.data[self.feat_name] < self.feat_value]
			mayores = self.data[self.data[self.feat_name] >= self.feat_value]

			menores = menores.drop(self.feat_name, 1)
			mayores = mayores.drop(self.feat_name, 1)

			if self.criterium == 'confianza':
				menores = menores.drop(self.feat_name + '_conf', 1)
				mayores = mayores.drop(self.feat_name + '_conf', 1)

			self.add_left(menores)
			self.add_right(mayores)

		# De lo contrario llamo a set_leaf para transformarlo en hoja
		else:
			self.set_leaf()

	# Busca el mejor corte posible para el nodo
	def split(self):
		# Inicializo la ganancia de info en el peor nivel posible
		max_gain = -float('inf')

		# Para cada feature (no considero la clase ni la completitud)
		filterfeatures = self.filterfeatures()

		for f in filterfeatures:

			# separo el dominio en todas las posibles divisiones para obtener la optima division
			pivotes = self.get_pivotes(self.data[f])

			for pivote in pivotes:

				# Separo las tuplas segun si su valor de esa variable es menor o mayor que el pivote
				menores = self.data[self.data[f] < pivote]
				mayores = self.data[self.data[f] >= pivote]

				# Calculo la ganancia de informacion para esta variable

				# gain = self.entropia - (len(menores) * self.entropy(menores) + len(mayores) * self.entropy(mayores)) / total
				if self.criterium == 'gain':
					gain = self.gain(menores, mayores)
				elif self.criterium == 'confianza':
					gain = self.confianza(menores, mayores, f)

				# Comparo con la ganancia anterior, si es mejor guardo el gain, la feature correspondiente y el pivote
				if(gain > max_gain):
					max_gain = gain
					self.feat_name = f
					self.feat_value = pivote

	def filterfeatures(self):
		# Para cada feature (no considero la clase ni la completitud)
		filterfeatures = []
		for feature in self.data.columns:
			if self.criterium == 'gain' and feature is not 'class':
				filterfeatures.append(feature)
			elif self.criterium == 'confianza' and not '_conf' in feature and feature is not 'class':
				filterfeatures.append(feature)
		return filterfeatures


	# determina se es necesario hacer un split de los datos
	def check_data(self):
		featuresfaltantes = self.filterfeatures()

		#if self.data['class'].nunique() == 1 or len(self.data.columns) == 1:
		if self.data['class'].nunique() == 1 or len(featuresfaltantes) == 0:
			return False
		else:
			return True

	# retorna una lista con los todos los threshold a evaluar para buscar la mejor separacion
	def get_pivotes(self, feature):

		#maximo = feature.max()
		#minimo = feature.min()
		#paso = (maximo - minimo) / 100

		#for i in range(100):
		#	pivotes.append( minimo + i*paso)

		return feature[1:].unique()

	# Convierte el nodo en hoja. Colocando la clase mas probable como resultado
	def set_leaf(self):
		self.is_leaf = True
		self.clase = stats.mode( self.data['class'])[0].item()

	def add_left(self, left_data):
		self.left = Node(left_data, self.criterium)

	def add_right(self, right_data):
		self.right = Node(right_data, self.criterium)

	def predict(self, tupla):
		if(self.is_leaf):
			return self.clase
		else:
			if(tupla['self.feat_name'] < self.feat_value):
				return self.left.predict(tupla)
			else:
				return self.right.predict(tupla)

	def gain(self, menores, mayores):

		total = len(self.data.index)

		gain = self.entropia - (len(menores) * self.entropy(menores) + len(mayores) * self.entropy(mayores)) / total

		return gain

	# Retorna la entropia de un grupo de datos
	def entropy(self, data):
		clases = data['class'].unique()
		total = len(data.index)

		entropia = 0

		for c in clases:
			p_c = len(data[data['class'] == c].index) / total
			entropia -= p_c * np.log2(p_c)

		return entropia


	def confianza(self, menores, mayores, feature):

		total = sum(menores[feature + '_conf']) + sum(mayores[feature + '_conf'])

		confianza = self.entropia - (sum(menores[feature + '_conf']) * self.trust(menores, feature) + sum(mayores[feature + '_conf']) * self.trust(mayores, feature)) / total

		return confianza

	# Retorna la completitud de un grupo de datos
	def trust(self, data, feature):

		clases = data['class'].unique()
		total = sum(data[feature + '_conf'])

		trust = 0

		for c in clases:
			p_c = sum(data[data['class'] == c][feature + '_conf']) / total
			trust -= p_c * np.log2(p_c)

		return trust