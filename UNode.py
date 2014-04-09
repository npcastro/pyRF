from node import *


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