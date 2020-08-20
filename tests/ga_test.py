import unittest
from genetic_algorithm.ga import SimpleGA
from genetic_algorithm.diversity_promoted_ga import DiversityPromotedGA
from genetic_algorithm.fitness_evaluator import FitnessEvaluator
from genetic_algorithm.diversity_evaluator import DiversityEvaluator
from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.network_schema import *
from memory.replay_memory import ReplayMemory
from unittest.mock import Mock
from utilities.pareto_solution import find_pareto

schema = {
            'conv': ConvSchema(1, 16, 3, 4),
            'fc': LinearSchema(128, 3)
        }

def is_offspring(a, b):
    num_similar_genes = 0
    for i in range(len(a.genes)):
        if torch.equal(a.genes[i], b.genes[i]):
            num_similar_genes += 1

    return num_similar_genes >= int(len(a.genes) / 2)

class GATest(unittest.TestCase):
    def test_simple_ga_gen_new_population(self):
        ga = SimpleGA(10, FitnessEvaluator(), selection_pressure=0.2, mutation_prob=0.01, mutation_power=1., crossover_prob=0.5)
        old_gen = [TensorGenotype(schema) for i in range(10)]
        fitnesses = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
        new_gen = ga.new_generation(old_gen, fitnesses)
        self.assertEqual(len(new_gen), ga.num_populations)
        self.assertTrue(old_gen[5] in new_gen) # Best individual survive
        
    def test_diversity_ga_gen_new_population(self):
        diversity = [0.5, 1.0, 1.0, 2.0, 2.5, 4.0, 4.0]
        diversity_evaluator = DiversityEvaluator()
        diversity_evaluator.eval_diversity = Mock(return_value=diversity)
        ga = DiversityPromotedGA(10, FitnessEvaluator, diversity_evaluator, mutation_prob=0.01, mutation_power=0., crossover_prob=0.5)
        old_gen = [TensorGenotype(schema) for i in range(10)]
        fitnesses = [6.0, 4.0, 1.0, 5.0, 2.0, 3.0, 1.0]
        new_gen = ga.new_generation(old_gen, fitnesses)
        self.assertEqual(len(new_gen), ga.num_populations)
        self.assertTrue(old_gen[0] in new_gen) # Best individual survive
        
        non_dominated = [old_gen[i] for i in find_pareto([fitnesses, diversity])]
        # New gen are children of pareto-efficient solutions
        for p in new_gen:
            is_non_dominated_child = [is_offspring(p, q) for q in non_dominated]
            self.assertTrue(any(is_non_dominated_child))
            