from node import *
import scipy.stats


class UNode(Node):
	def __init__(self, data, level=1, max_depth=8, min_samples_split=10):

		Node.__init__(self, data, level, max_depth, min_samples_split)

	# determina se es necesario hacer un split de los datos
	def check_data(self):
		featuresfaltantes = self.filterfeatures()
		if self.data['class'].nunique() == 1 or len(featuresfaltantes) == 0:
			return False
		elif self.level >= self.max_depth:
			return False
		# En este caso sumo las masa de probabilidad y la comparo con la minima necesaria para un split
		elif self.data['weight'].sum() < self.min_samples_split:
			return False
		else:
			return True

	# Creo que no es necesario que reciba el frame
	def total_samples_mass(data, clase):
		return data[data['class'] == clase]['weight'].sum()

	# retorna todos los limites derechos e izquierdos distintos de una feature
	def get_pivotes(self, feature, calidad = 'exact'):

		name = feature.name.rstrip('.mean')
		bounds = self.data[name + '.l'] + self.data[name + '.r']

		# Para eliminar valores repetidos
		return list(set(bounds))

	def get_menores(self, feature_name, pivote):
		menores = []

		# limpio el nombre de la feature
		feature_name = feature_name.rstrip('.mean')

		# Para cada tupla en el frame
		for index, row in self.data.iterrows():

			# si toda la masa de probabilidad esta bajo el pivote
			if row[feature_name + '.r'] <= pivote:
				menores.append(row)

			# si no hay masa de probabilidad bajo el pivote
			elif row[feature_name + '.l'] >= pivote:
				continue

			# si una fraccion de la masa esta bajo el pivote
			else:
				# obtengo los parametros de la distribucion de la feature
				menor = row
				w = menor['weight']
				mean = menor[feature_name+'.mean']
				std = menor[feature_name+'.std']
				l = menor[feature_name+'.l']
				r = menor[feature_name+'.r']


				# calculo el nuevo peso de tupla cuya feature es menor al pivote
				menor['weight'] = self.get_weight(w, mean , std, l, r, pivote, 'menor')

				# pongo la distribucion para la nueva tupla
				menor[feature_name + '.r'] = pivote
				menores.append(menor)

		return menores
		# return self.data[self.data[feature] < pivote]

	def get_mayores(self, feature, pivote):
		# return self.data[self.data[feature] >= pivote]
		mayores = []

		# limpio el nombre de la feature
		feature_name = feature_name.rstrip('.mean')

		# Para cada tupla en el frame
		for index, row in self.data.iterrows():

			# si toda la masa de probabilidad esta sobre el pivote
			if row[feature_name + '.l'] >= pivote:
				mayores.append(row)

			# si no hay masa de probabilidad sobre el pivote
			elif row[feature_name + '.r'] <= pivote:
				continue

			# si una fraccion de la masa esta sobre el pivote
			else:
				# obtengo los parametros de la distribucion de la feature
				mayor = row
				w = mayor['weight']
				mean = mayor[feature_name+'.mean']
				std = mayor[feature_name+'.std']
				l = mayor[feature_name+'.l']
				r = mayor[feature_name+'.r']

				# calculo el nuevo peso de la nueva tupla cuyo valor de la feature es mayor al pivote
				mayor['weight'] = self.get_weight(w, mean , std, l, r, pivote, 'mayor')

				# pongo la distribucion para la nueva tupla
				mayor[feature_name + '.r'] = pivote
				mayores.append(mayor)

		return mayores

	"""
	Determina la distribucion de probabilidad gaussiana acumulada entre dos bordes.
	mean: media de la gaussiana
	std: desviacion standard
	l: limite izquierdo
	r: limite derecho
	pivote: valor de corte
	how: determina si la probabilidad se calcula desde l hasta pivote o desde pivote hasta r
	"""
	def get_weight(w, mean, std, l, r, pivote, how='menor'):
		
		total_mass = scipy.stats.norm(mean, std).cdf(r) - scipy.stats.norm(mean, std).cdf(l)

		if how == 'menor':
			pivot_mass = scipy.stats.norm(mean, std).cdf(pivote) - scipy.stats.norm(mean, std).cdf(l)
			return w * (pivot_mass / total_mass)

		elif how == 'mayor':
			pivot_mass = scipy.stats.norm(mean, std).cdf(r) - scipy.stats.norm(mean, std).cdf(pivote)
			return w * (pivot_mass / total_mass)


	# Retorna la ganancia de dividir los datos en menores y mayores.
	# Deje la variable feature que no me sirve en la clase base, solo para ahorrarme repetir el metodo split. 
	# Eso debe poder arreglarse
	def gain(self, menores, mayores, feature):

		total = data['weight'].sum()

		gain = self.entropia - (menores['weight'].sum() * self.entropy(menores) + mayores['weight'].sum() * self.entropy(mayores)) / total

		return gain

	# Retorna la entropia de un grupo de datos
	def entropy(self, data):
		clases = data['class'].unique()

		# El total es la masa de probabilidad total del grupo de datos
		total = data['weight'].sum()

		entropia = 0

		for c in clases:
			p_c = data[data['class'] == c]['weight'].sum() / total
			entropia -= p_c * np.log2(p_c)

		return entropia