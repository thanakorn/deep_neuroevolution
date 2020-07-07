import unittest
import torch
import torch.nn as nn
from genetic_algorithm.genotype import NetworkGranularGenotype
from genetic_algorithm.chromosomes import *
from genetic_algorithm.opetators import *
from utilities.random_generator import RandomGenerator, NPRandomGenerator
from unittest.mock import Mock

schema = {
    'conv': ConvChromosome(3, 16, 3, 4),
    'fc1': LinearChromosome(128, 16),
    'fc2': LinearChromosome(16, 2)
}

    
class OpetatorTest(unittest.TestCase):
    
    def test_select_elites(self):
        genotypes = [NetworkGenotype(schema) for i in range(7)]
        fitnesses = [0.01, 1.0, 0.8, 0.2, 0.9, 0.75, 0.2]
        elites = select_elites(genotypes, fitnesses, 3)
        self.assertEqual(elites[0], genotypes[2])
        self.assertEqual(elites[1], genotypes[4])
        self.assertEqual(elites[2], genotypes[1])
    
    def test_mutation(self):
        parent = NetworkGranularGenotype(schema)
        random_generator = RandomGenerator()
        mutate_idx = [3, 25, 19]
        random_generator.randint = Mock(return_value=mutate_idx)
        child = mutate(parent, random_generator, mutation_rate=0.01)
        self.assertFalse(torch.equal(parent.genes[3], child.genes[3]))
        self.assertFalse(torch.equal(parent.genes[25], child.genes[25]))
        self.assertFalse(torch.equal(parent.genes[19], child.genes[19]))
        for i in range(len(parent.genes)):
            if i in mutate_idx: continue
            self.assertTrue(torch.equal(parent.genes[i], child.genes[i]))
            
    def test_num_genes_mutated(self):
        schema = {
            'conv1': ConvChromosome(3, 16, 7, 4),
            'conv2': ConvChromosome(16, 32, 3, 4),
            'fc1': LinearChromosome(1024, 64),
            'fc2': LinearChromosome(16, 2)
        }
        parent = NetworkGranularGenotype(schema)
        random_generator = NPRandomGenerator()
        mutation_rate = 0.02
        child = mutate(parent, random_generator, mutation_rate)
        num_mutated = 0
        for i in range(len(parent.genes)):
            if not torch.equal(parent.genes[i], child.genes[i]): num_mutated += 1
        self.assertEqual(num_mutated, int(mutation_rate * len(parent.genes)))
        
    def test_crossover(self):
        schema = {
            'conv': ConvChromosome(3, 4, 3, 4),
            'fc': LinearChromosome(128, 5),
            'out': LinearChromosome(5, 2)
        }
        a = NetworkGranularGenotype(schema)
        b = NetworkGranularGenotype(schema)
        random_generator = RandomGenerator()
        cross_idx = [3, 11, 9, 0, 1, 6, 5]
        random_generator.randint = Mock(return_value=cross_idx)
        c = crossover(a, b, random_generator)
        for i in range(len(a.genes)):
            if i in cross_idx: self.assertTrue(torch.equal(b.genes[i], c.genes[i]))
            else: self.assertTrue(torch.equal(a.genes[i], c.genes[i]))
            
    def test_num_genes_crossover(self):
        schema = {
            'conv': ConvChromosome(3, 4, 3, 4),
            'fc': LinearChromosome(128, 5),
            'out': LinearChromosome(5, 2)
        }
        a = NetworkGranularGenotype(schema)
        b = NetworkGranularGenotype(schema)
        random_generator = NPRandomGenerator()
        c = crossover(a, b, random_generator)
        num_genes_from_a = 0
        num_genes_from_b = 0
        for i in range(len(a.genes)):
            if torch.equal(a.genes[i], c.genes[i]): num_genes_from_a += 1
            elif torch.equal(b.genes[i], c.genes[i]): num_genes_from_b += 1
        self.assertEqual(int(0.5 * len(a.genes)), num_genes_from_a)
        self.assertEqual(int(0.5 * len(b.genes)), num_genes_from_b)
    
    def test_gen_population_mutation(self):
        parents = [NetworkGranularGenotype(schema) for i in range(5)]
        children = gen_population_mutation(parents, 50)
        self.assertEqual(len(children), 50)
        
    def test_gen_population_crossover(self):
        parents = [NetworkGranularGenotype(schema) for i in range(5)]
        children = gen_population_crossover(parents, 10)
        self.assertEqual(len(children), 10)