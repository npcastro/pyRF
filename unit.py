import unittest
from UNode import UNode


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

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSplittingMeasures)
    unittest.TextTestRunner(verbosity=2).run(suite)
