import unittest
from genetic_algorithm.ga import SimpleGA
from genetic_algorithm.fitness_evaluator import FitnessEvaluator
from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.chromosomes import *
from model.dqn import DQN

schema = {
    'conv': ConvChromosome(1, 16, 3, 4),
    'fc': LinearChromosome(128, 3)
}

class GATest(unittest.TestCase):
    def test_gen_new_population(self):
        ga = SimpleGA(10, FitnessEvaluator(), selection_pressure=0.2, mutation_prob=0.01, mutation_power=1., crossover_prob=0.5,
                      model_type=DQN, input_channels=4, img_size=84, num_actions=2)
        old_gen = [TensorGenotype(schema) for i in range(10)]
        fitnesses = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
        new_gen = ga.new_generation(old_gen, fitnesses)
        self.assertEqual(len(new_gen), ga.num_populations)
        self.assertTrue(old_gen[5] in new_gen) # Best individual survive
        self.assertTrue(True)