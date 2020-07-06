import unittest
from genetic_algorithm.ga import SimpleGA
from genetic_algorithm.fitness_evaluator import FitnessEvaluator
from genetic_algorithm.genotype import NetworkGenotype
from model.simple_cnn import SimpleCNN
from genetic_algorithm.chromosomes import *

schema = {
    'conv': ConvChromosome(1, 16, 3, 4),
    'fc': LinearChromosome(128, 3)
}

class GATest(unittest.TestCase):
    def test_gen_new_population(self):
        ga = SimpleGA(SimpleCNN, 10, FitnessEvaluator(), selection_pressure=0.2)
        old_gen = [NetworkGenotype(schema) for i in range(10)]
        fitnesses = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
        new_gen = ga.new_generation(old_gen, fitnesses)
        self.assertEqual(len(new_gen), ga.num_populations)
        self.assertTrue(old_gen[5] in new_gen) # Best individual survive
        self.assertTrue(True)