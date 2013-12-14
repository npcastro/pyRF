

# data es un dataframe que tiene que contener una columna class. La cual el arbol intenta predecir.
# podria pensar en relajar esto y simplemente indicar cual es la variable a predecir.

class Node:

	def __init__(self, data):

		self.data = data
		self.is_leaf = False
		self.feat_name = ""
		self.feat_value = None
		self.left = None
		self.right = None

	def split(self):
		pass

	def set_leaf(self, is_leaf):
		pass

	def add_left(self, left_data):
		self.left = Node(left_data)

	def add_right(self, right_data):
		self.right = Node(right_data)