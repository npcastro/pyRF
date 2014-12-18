import unittest
from UNode import UNode


class TestSplittingMeasures(unittest.TestCase):

    # Entropia
    def setUp(self):
        self.unode = UNode(None)

    # Happy path
    def test_normal_case(self):
        pass

    # Testear valores cero

    # Testear valores negativos

    # Testear maximo

    # Testear minimo

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSplittingMeasures)
    unittest.TextTestRunner(verbosity=2).run(suite)
