from node import *

class Tree:

	def __init__(self, criterium):
		self.root = []
		self.criterium = criterium		# "gain"

	# recibe un set de entrenamiento y ajusta el arbol
	def fit(self, data):
		self.root = Node(data)

	# recibe un dato y retorna prediccion
	def predict(self):
		pass

	# seria bueno poder ver la estructura del arbol. 
	def show(self):
		pass