import unittest
from genetic_algorithm.crowding_ga import DeterministicCrowdingGA
from genetic_algorithm.fitness_evaluator import FitnessEvaluator
from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.network_schema import *
from genetic_algorithm.opetators import mutate
from utilities.random_generator import NPRandomGenerator

schema = {
    'fc': LinearSchema(10, 5)
}
random_generator = NPRandomGenerator()

def is_offspring(a, b):
    num_similar_genes = 0
    for i in range(len(a.genes)):
        if torch.equal(a.genes[i], b.genes[i]):
            num_similar_genes += 1

    return num_similar_genes == int(len(a.genes) / 2)

class CrowdingGATest(unittest.TestCase):
    def test_gen_offsprings(self):
        num_population = 4
        ga = DeterministicCrowdingGA(num_population, FitnessEvaluator, selection_pressure=1.0, mutation_prob=1.0, mutation_power=0., crossover_prob=0.5)
        parents = [mutate(TensorGenotype(schema), random_generator, 1., 0.1) for i in range(num_population)]
        offsprings = ga.gen_offsprings(parents)
        
        # Offspring_i has to be a child of parent_i
        for i in range(num_population):
            self.assertTrue(is_offspring(parents[i], offsprings[i]))
            
    def test_replace_parents(self):
        num_population = 4
        ga = DeterministicCrowdingGA(num_population, FitnessEvaluator, selection_pressure=1.0, mutation_prob=1.0, mutation_power=0., crossover_prob=0.5)
        parents = [TensorGenotype(schema) for i in range(num_population)]
        parent_fitnesses = [0.1, 0.5, 0.7, 0.2]
        offsprings = ga.gen_offsprings(parents)
        offspring_fitnesses = [0.15, 0.3, 0.8, 0.1]
        
        new_gen = ga.replace_parents(parents, parent_fitnesses, offsprings, offspring_fitnesses)
        self.assertEqual(new_gen[0], offsprings[0])
        self.assertEqual(new_gen[1], parents[1])
        self.assertEqual(new_gen[2], offsprings[2])
        self.assertEqual(new_gen[3], parents[3])
        