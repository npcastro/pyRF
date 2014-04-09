from node import *

class CompNode(Node):
    def __init__(self, data, level=1, max_depth=8, min_samples_split=10):

        Node.__init__(self, data, level, max_depth, min_samples_split)

    def gain(self, menores, mayores, feature):
        total = sum(menores[feature + '_comp']) + sum(mayores[feature + '_comp'])

        confianza = self.entropia - (
            sum(menores[feature + '_comp']) * self.trust(menores, feature) + sum(
                mayores[feature + '_comp']) * self.trust(
                mayores, feature)) / total

        return confianza

    # Retorna la entropia, calculada con confianza, de un grupo de datos en una variable.
    def trust(self, data, feature):

        clases = data['class'].unique()
        total = sum(data[feature + '_comp'])

        trust = 0

        for c in clases:
            p_c = sum(data[data['class'] == c][feature + '_comp']) / total
            trust -= p_c * np.log2(p_c)

        return trust

    def predict(self, tupla, confianza=1):
        if self.is_leaf:
            return self.clase, confianza
        else:
            if tupla[self.feat_name] < self.feat_value:
                # Propago la incertidumbre del dato que estoy prediciendo
                # return self.left.predict(tupla, confianza * tupla[self.feat_name + '_comp'])
                return self.left.predict(tupla, (confianza + tupla[self.feat_name + '_comp'])/2)
                
            else:
                # return self.right.predict(tupla, confianza * tupla[self.feat_name + '_comp'])
                return self.right.predict(tupla, (confianza + tupla[self.feat_name + '_comp'])/2)