from node import *
#import scipy.stats
import pandas as pd
from copy import deepcopy
import pyRF_prob


class UNode(Node):
	def __init__(self, data, level=1, max_depth=8, min_samples_split=10):

		Node.__init__(self, data, level, max_depth, min_samples_split)
		self.mass = int(self.data['weight'].sum())			

	def get_pivotes(self, feature, calidad = 'exact'):
		"""
		Retorna todos los valores segun los que se debe intentar cortar una feature
		"""
		name = feature.name.rstrip('.mean')
		bounds = self.data[name + '.l'].tolist() + self.data[name + '.r'].tolist()


		ret = list(set(bounds)) # Para eliminar valores repetidos
		ret.sort()	# Elimino los bordes, aunque talvez sea mejor poner un if mas adelante noma
		return ret[1:-1]

	# Busca el mejor corte posible para el nodo
	def split(self):

		# Inicializo la ganancia de info en el peor nivel posible
		max_gain = -float('inf')

		# Para cada feature (no considero la clase ni la completitud)
		filterfeatures = self.filterfeatures()

		print filterfeatures

		for f in filterfeatures:

			# Limpio el nombre de la feature
			feature_name = f.rstrip('.mean')
			print 'Evaluando feature: ' + f

			# Ordeno el frame segun la media de la variable
			data_por_media = self.data.sort(f, inplace=False)

			#Transformo la informacion relevante de esta feature a listas
			w_list = data_por_media['weight'].tolist()
			mean_list = data_por_media[feature_name + '.mean'].tolist()
			std_list = data_por_media[feature_name + '.std'].tolist()
			left_bound_list = data_por_media[feature_name + '.l'].tolist()
			right_bound_list = data_por_media[feature_name + '.r'].tolist()
			class_list = data_por_media['class'].tolist()

			menores_index = 0
			mayores_index = 0

			old_menores_index = 0
			old_mayores_index = 0

			# Obtengo las clases existentes
			clases = list(set(class_list))

			# Creo diccionarios para guardar la masa de los estrictos menores y estrictos mayores, y asi no calcularla continuamente.
			# Los menores parten vacios y los mayores parten con toda la masa
			menores_estrictos_mass = { c: 0 for c in clases}
			mayores_estrictos_mass = data_por_media.groupby('class')['weight'].sum().to_dict()

			# Me muevo a traves de los posibles pivotes
			for i in xrange(1,self.n_rows):

				pivote = data_por_media.at[i,f]

				# Actualizo los indices
				menores_index, mayores_index = self.update_indexes(menores_index, mayores_index, pivote, left_bound_list, right_bound_list)
				print menores_index, mayores_index

				# Actualizo la masa estrictamente menor y mayor
				for i in xrange(old_menores_index, menores_index):
					menores_estrictos_mass[class_list[i]] += w_list[i]

				for i in xrange(old_mayores_index, mayores_index):
					mayores_estrictos_mass[class_list[i]] -= w_list[i]

				old_menores_index, old_mayores_index = menores_index, mayores_index

				# menores, mayores = self.split_tuples_by_pivot(w_list, mean_list, std_list, left_bound_list, right_bound_list, class_list, pivote)
				w_list_afectada = w_list[menores_index:mayores_index]
				mean_list_afectada = mean_list[menores_index:mayores_index]
				std_list_afectada = std_list[menores_index:mayores_index]
				left_bound_list_afectada = left_bound_list[menores_index:mayores_index]
				right_bound_list_afectada = right_bound_list[menores_index:mayores_index]
				class_list_afectada = class_list[menores_index:mayores_index]

				menores, mayores = self.split_tuples_by_pivot(w_list_afectada, mean_list_afectada, std_list_afectada, left_bound_list_afectada, right_bound_list_afectada, class_list_afectada, pivote, menores_estrictos_mass, mayores_estrictos_mass)

				# No se si es necesario
				if not any(menores) or not any(mayores):
					return

				# Calculo la ganancia de informacion para esta variable
				pivot_gain = self.gain(menores, mayores)
				
				if pivot_gain > max_gain:
					max_gain = pivot_gain
					self.feat_value = pivote
					self.feat_name = feature_name + '.mean'				

			break # Para testear cuanto se demora una sola feature

	# Toma los indices de los estrictamente menores y mayores, mas el nuevo pivote y los actualiza
	def update_indexes(self, menores_index, mayores_index, pivote, limites_l, limites_r):

		# Actualizo menores
		ultimo_r_menor = limites_r[menores_index]

		# Itero hasta encontrar una tupla que NO sea completamente menor que el pivote
		while( ultimo_r_menor <= pivote):
			menores_index += 1
			ultimo_r_menor = limites_r[menores_index]

		# Actualizo mayores
		ultimo_l_mayor = limites_l[mayores_index]

		# Itero hasta encontrar una tupla que SEA completamente mayor que el pivote
		while( ultimo_l_mayor <= pivote and mayores_index < len(limites_l) - 1):
			ultimo_l_mayor = limites_l[mayores_index]
			mayores_index += 1

		return menores_index, mayores_index


	def split_tuples_by_pivot(self, w_list, mean_list, std_list, left_bound_list, right_bound_list, class_list, pivote, menores, mayores):
		"""
		Toma un grupo de datos lo recorre entero y retorna dos diccionarios con las sumas de masa 
		separadas por clase. Un diccionario es para los datos menores que el pivote y el otro para los mayores
		"""

		# Obtengo las clases existentes
		# clases = list(set(class_list))

		# # Creo diccionarios para guardar
		# menores = { c: 0 for c in clases}
		# mayores = { c: 0 for c in clases}

		for i in xrange(len(class_list)):

			# if right_bound_list[i] <= pivote:
			# 	menores[class_list[i]] += w_list[i]

			# elif left_bound_list[i] >= pivote:
			# 	mayores[class_list[i]] += w_list[i]

			# else:
				# mass_menor, mass_mayor = self.split_tuple( w_list[i], mean_list[i], std_list[i], left_bound_list[i], right_bound_list[i], pivote)

				# menores[class_list[i]] += mass_menor
				# mayores[class_list[i]] += mass_mayor	
			mass_menor, mass_mayor = self.split_tuple( w_list[i], mean_list[i], std_list[i], left_bound_list[i], right_bound_list[i], pivote)
			menores[class_list[i]] += mass_menor
			mayores[class_list[i]] += mass_mayor
			
		return menores, mayores

	# Toma una sola tupla y la corta segun pivote retornando el pedazo mayor y el menor
	def split_tuple(self, w, mean, std, left_bound, right_bound, pivote):

		#Estamos seguros que esto esta bien?? 

		# Corto la parte de la tupla menor que el pivote
		masa_menor = w * pyRF_prob.cdf(pivote, mean, std, left_bound, right_bound)

		# Corte la parte de la tupla mayor que el pivote
	 	masa_mayor = w * (1 - pyRF_prob.cdf(pivote, mean, std, left_bound, right_bound))

	 	return masa_menor, masa_mayor


	def get_menores(self, feature_name, pivote):
		menores = []

		# limpio el nombre de la feature
		feature_name = feature_name.rstrip('.mean')

		# menores = self.data[self.data[feature_name + '.l'] < pivote]
		self.data.apply(func=self.get_weight, axis=1, args=[menores, pivote, feature_name, "menor"])

		return pd.DataFrame(menores)


	def get_mayores(self, feature_name, pivote):
		mayores = []

		# limpio el nombre de la feature
		feature_name = feature_name.rstrip('.mean')

		# mayores = self.data[self.data[feature_name + '.r'] >= pivote]
		self.data.apply(func=self.get_weight, axis=1, args=[mayores, pivote, feature_name, "mayor"])

		return pd.DataFrame(mayores)


	def get_weight(self, tupla, lista, pivote, feature_name, how):
		"""
		Determina la distribucion de probabilidad gaussiana acumulada entre dos bordes.
		
		pivote: valor de corte
		how: determina si la probabilidad se calcula desde l hasta pivote o desde pivote hasta r
		"""

		left_bound = tupla[feature_name+'.l']
		right_bound = tupla[feature_name+'.r']

		if how == 'menor' and pivote <= left_bound or how == 'mayor' and pivote >= right_bound:
		 	return

		elif left_bound >= pivote and how == 'mayor' or right_bound <= pivote and how == 'menor':
			lista.append(tupla)
			return

		else:
			w = tupla['weight']
			mean = tupla[feature_name+'.mean']
			std = tupla[feature_name+'.std']
			
			if how == 'menor':
				tupla['weight'] = min(w * pyRF_prob.cdf(pivote, mean, std, left_bound, right_bound), 1)
				# tupla[feature_name+'.r'] = min(pivote, tupla[feature_name + '.r'])
				tupla[feature_name+'.r'] = pivote
				lista.append(tupla)
				return

			elif how == 'mayor':
			 	tupla['weight'] = min(w * (1 - pyRF_prob.cdf(pivote, mean, std, left_bound, right_bound)), 1)
			 	# tupla[feature_name+'.l'] = max(pivote, tupla[feature_name + '.l'])
			 	tupla[feature_name+'.l'] = pivote
			 	lista.append(tupla)
				return


	def gain(self, menores, mayores):
		"""
			Retorna la ganancia de dividir los datos en menores y mayores
			Menores y mayores son diccionarios donde la llave es el nombre de la clase y los valores son la suma de masa para ella.
		"""
		gain = self.entropia - ( sum(menores.values()) * self.entropy(menores) + sum(mayores.values()) * self.entropy(mayores) ) / self.mass

		return gain

	# Retorna la ganancia de dividir los datos en menores y mayores.
	# Deje la variable feature que no me sirve en la clase base, solo para ahorrarme repetir el metodo split. 
	# Eso debe poder arreglarse
	def gain_old(self, menores, mayores, feature):

		# total = self.data['weight'].sum()

		# gain = self.entropia - (menores['weight'].sum() * self.entropy(menores) + mayores['weight'].sum() * self.entropy(mayores)) / total
		gain = self.entropia - (menores['weight'].sum() * self.entropy(menores) + mayores['weight'].sum() * self.entropy(mayores)) / self.mass

		return gain

	def entropy(self, data):
		"""
		Retorna la entropia de un grupo de datos.
		data: diccionario donde las llaves son nombres de clases y los valores sumas (o conteos de valores)
		"""

		total = sum(data.values())
		entropia = 0
		
		for clase in data.keys():
			entropia -= (data[clase] / total) * np.log(data[clase] / total)

		return entropia

	# Retorna la entropia de un grupo de datos
	def entropy_old(self, data):

		# El total es la masa de probabilidad total del grupo de datos
		total = data['weight'].sum()
		log = np.log2
		entropia = 0

		pesos = data.groupby('class')['weight']
		for suma in pesos.sum():
			entropia -= (suma / total) * log(suma / total)

		return entropia

	def predict(self, tupla, prediction={}, w=1):
		# Si es que es el nodo raiz
		if len(prediction.keys()) == 0:
			prediction = {c: 0.0 for c in self.data['class'].unique() }

		if self.is_leaf:
			aux = deepcopy(prediction)
			aux[self.clase] += w
			return aux

		# Puede que falte chequear casos bordes, al igual que lo hago en get_menores y get_mayores
		else:
			feature_name = self.feat_name.rstrip('.mean')
			mean = tupla[feature_name + '.mean']
			std = tupla[feature_name + '.std']
			l = tupla[feature_name + '.l']
			r = tupla[feature_name + '.r']
			pivote = self.feat_value

			w_right = self.get_weight(w, mean, std, l, r, pivote, 'mayor')
			w_left = self.get_weight(w, mean, std, l, r, pivote, 'menor')
			
			a = self.right.predict(tupla, prediction, w_right)
			b = self.left.predict(tupla, prediction, w_left)

			# Tengo que retornar la suma elementwise de los diccionarios a y b
			return {key: a[key] + b[key] for key in a}