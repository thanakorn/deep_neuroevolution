import unittest
from utilities.pareto_solution import find_pareto

class ParetoTest(unittest.TestCase):
    def test_find_pareto_solutions(self):
        u1 = [6.0, 4.0, 1.0, 5.0, 2.0, 3.0, 1.0]
        u2 = [0.5, 1.0, 1.0, 2.0, 2.5, 4.0, 4.0]
        utilities = [u1, u2]
        
        pareto_idx = find_pareto(utilities)
        self.assertEqual(len(pareto_idx), 4)
        self.assertTrue(0 in pareto_idx)
        self.assertTrue(3 in pareto_idx)
        self.assertTrue(5 in pareto_idx)
        self.assertTrue(6 in pareto_idx)