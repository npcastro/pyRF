import unittest
from UNode import UNode
from node import Node
import pyRF_prob
import tree
import pandas as pd


class TestProbabilityMethods(unittest.TestCase):
    def setUp(self):
        pass

    def test_split_at_left_border(self):
        # pyRF_prob.cdf(pivote, mean, std, left_bound, right_bound)
        feature_mass = pyRF_prob.cdf(2, 5, 1, 2, 8)
        self.assertEqual(feature_mass, 0)

        feature_mass = pyRF_prob.cdf(5, 5, 1, 5, 8)
        self.assertEqual(feature_mass, 0)

        feature_mass = pyRF_prob.cdf(6, 5, 1, 6, 8)
        self.assertEqual(feature_mass, 0)

    def test_split_at_right_border(self):
        feature_mass = pyRF_prob.cdf(8, 5, 1, 2, 8)
        self.assertEqual(feature_mass, 1)

        feature_mass = pyRF_prob.cdf(5, 5, 1, 2, 5)
        self.assertEqual(feature_mass, 1)

        feature_mass = pyRF_prob.cdf(4, 5, 1, 2, 4)
        self.assertEqual(feature_mass, 1)

    # def test_normal_case(self):
    #     pass


class TestSplittingMethods(unittest.TestCase):

    def setUp(self):
        self.unode = UNode(None)

    def test_normal_case(self):
        left_values = [-8, -7, -5, -3, 0, 2, 3.5, 6]
        right_values = [-6, -4, -1, 1, 2.5, 4, 5, 7]
        clases = ['a', 'b', 'b', 'b', 'a', 'b', 'a', 'b']

        bounds = self.unode.get_class_changes(left_values, right_values, clases)
        self.assertEqual(set(bounds), set([-7, -6, 0, 1, 2, 2.5, 3.5, 4, 5]))

    # numeros negativos

    # puntos iguales

    # multiples clases

# testear que el argsort esta funcionando correctamente?


class TestClassDistribution(unittest.TestCase):

    def setUp(self):
        self.node = Node(None)

    def test_normal_case(self):
        classes = ['a', 'a', 'b', 'a', 'c', 'c']
        test = self.node.get_class_distribution(classes)

        self.assertEqual(test['a'], 3)
        self.assertEqual(test['b'], 1)
        self.assertEqual(test['c'], 2)


class TestSplittingMeasures(unittest.TestCase):

    # Entropia
    def setUp(self):
        self.unode = UNode(None)

    def test_normal_case(self):
        values = {'a': 5.0, 'b': 15.0, 'c': 0.0}
        self.assertEqual(self.unode.entropy(values), 0.81127812445913283)

    def test_integer_values(self):
        values = {'a': 5, 'b': 15, 'c': 0}
        self.assertEqual(self.unode.entropy(values), 0.81127812445913283)

    # Testear maximo
    def test_maximum_entropy(self):
        values = {'a': 5, 'b': 5, 'c': 5, 'd': 5}
        self.assertEqual(self.unode.entropy(values), 2)

    # Testear minimo
    def test_minimum_entropy(self):
        values = {'a': 10, 'b': 0, 'c': 0, 'd': 0}
        self.assertEqual(self.unode.entropy(values), 0)

    # Testear valores negativos?

class TestFeatureSelection(unittest.TestCase):

    def setUp(self):
        data = pd.read_csv('sets/iris viejo/iris.data', sep=',', header=None,
                           names=['sepal length', 'sepal width', 'petal length',
                           'petal width', 'class'])
        y = data['class']
        data = data.drop('class', axis=1)

        self.clf = tree.Tree('gain')
        self.clf.fit(data, y)

    def test_get_splits(self):

        # Diccionario con los puntos de corte
        test_splits = {'petal length': [3.0, 4.9, 5.0], 'petal width': [1.8, 1.7]}
        splits = self.clf.get_splits()
        self.assertEqual(splits, test_splits)

    # def test_select_feats(self):
        # Features importantes
        # test_feats = ['petal length', 'petal width']


class TestParallel(unittest.TestCase):

    def setUp(self):
        data = pd.read_csv('sets/iris random/iris random 25.csv')
        data = data.dropna(axis=0, how='any')
        data['weight'] = data['weight'].astype(float)

        self.y = data['class']
        self.data = data.drop('class', axis=1)

        self.clf_normal = tree.Tree('uncertainty', max_depth=12,
                        min_samples_split=10, most_mass_threshold=0.99, min_mass_threshold=0.10,
                        min_weight_threshold=0.01)

        self.clf_parallel = tree.Tree('uncertainty', max_depth=12,
                        min_samples_split=10, most_mass_threshold=0.99, min_mass_threshold=0.10,
                        min_weight_threshold=0.01, parallel = True)

    def test_same_result(self):

        self.clf_normal.fit(self.data, self.y)
        self.clf_parallel.fit(self.data, self.y)

        splits_normal = self.clf_normal.get_splits()
        splits_parallel = self.clf_parallel.get_splits()
        self.assertEqual(splits_normal, splits_parallel)

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestProbabilityMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestSplittingMeasures))
    suite.addTests(loader.loadTestsFromTestCase(TestClassDistribution))
    suite.addTests(loader.loadTestsFromTestCase(TestSplittingMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestParallel))

    unittest.TextTestRunner(verbosity=2).run(suite)
