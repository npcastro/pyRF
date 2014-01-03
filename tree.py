from node import *
import pandas as pd

class Tree:

	def __init__(self, criterium):
		self.root = []
		self.criterium = criterium

	# recibe un set de entrenamiento y ajusta el arbol
	def fit(self, data):
		self.root = Node(data, self.criterium)

	# recibe un dato y retorna prediccion
	def predict(self, tupla):
		return self.root.predict(tupla)

	# recibe un frame completo y retorna otro frame con la clase original, la predicha y la confianza de la prediccion
	def predict_table(self, frame):

		# Creo el frame e inserto los resultados
		tabla = []
		for index, row in frame.iterrows():
			clase = row['class']
			predicted, confianza = self.root.predict(row)
			tabla.append([clase, predicted, confianza])

		return pd.DataFrame(tabla, index=frame.index)

	# seria bueno poder ver la estructura del arbol
	def show(self):
		pass
