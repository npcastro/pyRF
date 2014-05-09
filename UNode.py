from node import *
#import scipy.stats
import pandas as pd
from copy import deepcopy
import pyRF_prob


class UNode(Node):
	def __init__(self, data, level=1, max_depth=8, min_samples_split=10):

		Node.__init__(self, data, level, max_depth, min_samples_split)
		self.mass = int(self.data['weight'].sum())			

	# Creo que no es necesario que reciba el frame
	def total_samples_mass(data, clase):
		return data[data['class'] == clase]['weight'].sum()

	# retorna todos los limites derechos e izquierdos distintos de una feature
	def get_pivotes(self, feature, calidad = 'exact'):

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

			# Hago tres copias del frame ordenadas por mean, l y r
			data_por_media = self.data.sort(f, inplace=False)

			# En este caso es lo mismo ordenar de las tres maneras. De todas formas parece que 
			# tendriamos que asumir simetria (gaussiana), para usar el approach que tenemos. 
			# data_por_l = self.data.sort(feature_name + '.l', inplace=False)
			# data_por_r = self.data.sort(feature_name + '.r', inplace=False)

			menores_index = 0
			mayores_index = 0

			# Me muevo a traves de los posibles pivotes
			for i in xrange(1,self.n_rows):

				pivote = data_por_media.at[i,f]

				# print pivote, menores_index, mayores_index, feature_name, max_gain
				self.bad_method(pivote, menores_index, mayores_index, feature_name, max_gain, data_por_media)

				# # Actualizo los indices. Tal vez se podria hacer por referencia. No creo que haga mucha diferencia
				# menores_index, mayores_index = self.update_indexes(menores_index, mayores_index, pivote, feature_name)

				# print mayores_index - menores_index

				# # Separo las tuplas completamente mayores o menores que los indices (no afectadas por pivote)
				# menores = data_por_media[0:menores_index]
				# mayores = data_por_media[mayores_index:]

				# # Separo las tuplas cortadas por el pivote
				# tuplas_afectadas_por_pivote = data_por_media[menores_index:mayores_index]
				
				# # Faltan un metodo split_tuple_by_pivot. Que tome por referencia menores, mayores, el pivote
				# # y las tuplas afectadas por el pivote y les agregue los pedazos de las tuplas cortadas.
				# menores, mayores = self.split_tuples_by_pivot(tuplas_afectadas_por_pivote, menores, mayores, pivote, feature_name)

				# # No se si es necesario
				# if menores.empty or mayores.empty:
				# 	continue

				# # Calculo la ganancia de informacion para esta variable
				# pivot_gain = self.gain(menores, mayores, f)

				# if pivot_gain > max_gain:
				# 	max_gain = pivot_gain
				# 	self.feat_value = pivote
				# 	self.feat_name = f

			break #para probar cuanto demora una sola feature

	def bad_method(self, pivote, menores_index, mayores_index, feature_name, max_gain, data_por_media):
		# Actualizo los indices. Tal vez se podria hacer por referencia. No creo que haga mucha diferencia
		menores_index, mayores_index = self.update_indexes(menores_index, mayores_index, pivote, feature_name, data_por_media)

		# Separo las tuplas completamente mayores o menores que los indices (no afectadas por pivote)
		menores = data_por_media[0:menores_index]
		mayores = data_por_media[mayores_index:]

		print menores_index
		print mayores_index

		# Separo las tuplas cortadas por el pivote
		tuplas_afectadas_por_pivote = data_por_media[menores_index:mayores_index]
		
		# Faltan un metodo split_tuple_by_pivot. Que tome por referencia menores, mayores, el pivote
		# y las tuplas afectadas por el pivote y les agregue los pedazos de las tuplas cortadas.
		self.split_tuples_by_pivot(tuplas_afectadas_por_pivote, menores, mayores, pivote, feature_name)

		# No se si es necesario
		if menores.empty or mayores.empty:
			return

		# Calculo la ganancia de informacion para esta variable
		pivot_gain = self.gain(menores, mayores, feature_name + '.mean')

		if pivot_gain > max_gain:
			max_gain = pivot_gain
			self.feat_value = pivote
			self.feat_name = feature_name + '.mean'

		return
	
	# Toma los indices de los estrictamente menores y mayores, mas el nuevo pivote y los actualiza
	def update_indexes(self, menores_index, mayores_index, pivote, feature_name, data):
		
		# Actualizo menores
		tupla = data.iloc[menores_index]

		# Itero hasta encontrar una tupla que NO sea completamente menor que el pivote
		while( tupla[feature_name + '.r'] <= pivote):
			menores_index += 1
			tupla = data.iloc[menores_index] 

		
		# Actualizo mayores
		tupla = data.iloc[mayores_index]

		
		# Itero hasta encontrar una tupla que SEA completamente mayor que el pivote
		while( tupla[feature_name + '.l'] <= pivote):
			mayores_index += 1
			tupla = data.iloc[mayores_index]
			

		return menores_index, mayores_index

	def split_tuples_by_pivot(self, tuplas_afectadas_por_pivote, menores, mayores, pivote, feature_name):
		aux_menores = []
		aux_mayores = []

		for index, row in tuplas_afectadas_por_pivote.iterrows():
			row_menor, row_mayor = self.split_tuple(row, pivote, feature_name)

			aux_menores.append(row_menor)
			aux_mayores.append(row_mayor)

		menores.append(aux_menores, ignore_index=False)
		mayores.append(aux_mayores, ignore_index=False)

		return 

	# Toma una sola tupla y la corta segun pivote retornando el pedazo mayor y el menor
	def split_tuple(self, tupla, pivote, feature_name):
		
		# Sera mejor pedir los parametros ya arreglados? onda peso, l, r etc.. O hacerlo como lo estamos haciendo ahora	
		w = tupla['weight']
		mean = tupla[feature_name + '.mean']
		std = tupla[feature_name + '.std']
		left_bound = tupla[feature_name + '.l']
		right_bound = tupla[feature_name + '.r']
		
		tupla_menor = tupla
		tupla_mayor = tupla

		# Corto la parte de la tupla menor que el pivote
		tupla_menor['weight'] = min(w * pyRF_prob.cdf(pivote, mean, std, left_bound, right_bound), 1)
		# tupla_menor[feature_name+'.r'] = min(pivote, tupla[feature_name + '.r'])
		tupla_menor[feature_name+'.r'] = pivote

		# Corte la parte de la tupla mayor que el pivote
	 	tupla_mayor['weight'] = min(w * (1 - pyRF_prob.cdf(pivote, mean, std, left_bound, right_bound)), 1)
	 	# tupla_menor[feature_name+'.l'] = max(pivote, tupla[feature_name + '.l'])
	 	tupla_mayor[feature_name+'.l'] = pivote

	 	return tupla_menor, tupla_mayor
		


	def get_menores_old(self, feature_name, pivote):
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
				menor['weight'] = self.get_weight_old(w, mean , std, l, r, pivote, 'menor')

				# pongo la distribucion para la nueva tupla
				menor[feature_name + '.r'] = pivote
				menores.append(menor)

		return pd.DataFrame(menores)
		return self.data[self.data[feature] < pivote]

	def get_mayores_old(self, feature_name, pivote):
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
				mayor['weight'] = self.get_weight_old(w, mean , std, l, r, pivote, 'mayor')

				# pongo la distribucion para la nueva tupla
				mayor[feature_name + '.r'] = pivote
				mayores.append(mayor)

		return pd.DataFrame(mayores)

	
	def get_weight_old(self, w, mean, std, l, r, pivote, how='menor'):
		"""
		Determina la distribucion de probabilidad gaussiana acumulada entre dos bordes.
		mean: media de la gaussiana
		std: desviacion standard
		l: limite izquierdo
		r: limite derecho
		pivote: valor de corte
		how: determina si la probabilidad se calcula desde l hasta pivote o desde pivote hasta r
		"""
		if how == 'menor' and pivote <= l or how == 'mayor' and pivote >= r:
			return 0

		# total_mass = scipy.stats.norm(mean, std).cdf(r) - scipy.stats.norm(mean, std).cdf(l)

		if how == 'menor':
		# 	pivot_mass = scipy.stats.norm(mean, std).cdf(pivote) - scipy.stats.norm(mean, std).cdf(l)
		#	return min([w * (pivot_mass / total_mass), 1])
			return min(w * pyRF_prob.cdf(pivote, mean, std, l, r), 1)

		elif how == 'mayor':
		# 	pivot_mass = scipy.stats.norm(mean, std).cdf(r) - scipy.stats.norm(mean, std).cdf(pivote)
		#	return min([w * (pivot_mass / total_mass), 1])
		 	return min(w * (1 - pyRF_prob.cdf(pivote, mean, std, l, r)), 1)
	

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
				# pivot_mass = scipy.stats.norm(mean, std).cdf(pivote) - scipy.stats.norm(mean, std).cdf(l)
				# return min([w * (pivot_mass / total_mass), 1])
				tupla['weight'] = min(w * pyRF_prob.cdf(pivote, mean, std, left_bound, right_bound), 1)
				# tupla[feature_name+'.r'] = min(pivote, tupla[feature_name + '.r'])
				tupla[feature_name+'.r'] = pivote
				lista.append(tupla)
				return

			elif how == 'mayor':
			# 	pivot_mass = scipy.stats.norm(mean, std).cdf(r) - scipy.stats.norm(mean, std).cdf(pivote)
			#	return min([w * (pivot_mass / total_mass), 1])
			 	tupla['weight'] = min(w * (1 - pyRF_prob.cdf(pivote, mean, std, left_bound, right_bound)), 1)
			 	# tupla[feature_name+'.l'] = max(pivote, tupla[feature_name + '.l'])
			 	tupla[feature_name+'.l'] = pivote
			 	lista.append(tupla)
				return


	# Retorna la ganancia de dividir los datos en menores y mayores.
	# Deje la variable feature que no me sirve en la clase base, solo para ahorrarme repetir el metodo split. 
	# Eso debe poder arreglarse
	def gain(self, menores, mayores, feature):

		# total = self.data['weight'].sum()

		# gain = self.entropia - (menores['weight'].sum() * self.entropy(menores) + mayores['weight'].sum() * self.entropy(mayores)) / total
		gain = self.entropia - (menores['weight'].sum() * self.entropy(menores) + mayores['weight'].sum() * self.entropy(mayores)) / self.mass

		return gain

	# Retorna la entropia de un grupo de datos
	def entropy(self, data):

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