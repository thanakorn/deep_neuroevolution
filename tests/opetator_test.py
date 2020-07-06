import unittest
import torch
import torch.nn as nn
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.chromosomes import *
from genetic_algorithm.opetators import *

schema = {
    'conv': ConvChromosome(1, 16, 3, 4),
    'fc': LinearChromosome(128, 3)
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
        parent = NetworkGenotype(schema)
        child = mutate(parent, mutation_power=1.)
        self.assertFalse(torch.equal(parent.chromosomes['conv.weight'], child.chromosomes['conv.weight']))
        self.assertFalse(torch.equal(parent.chromosomes['conv.bias'], child.chromosomes['conv.weight']))
        self.assertFalse(torch.equal(parent.chromosomes['fc.weight'], child.chromosomes['fc.weight']))
        self.assertFalse(torch.equal(parent.chromosomes['fc.bias'], child.chromosomes['fc.weight']))
        
    def test_gen_population_mutation(self):
        parents = [NetworkGenotype(schema) for i in range(5)]
        children = gen_population_mutation(parents, 50)
        self.assertEqual(len(children), 50)
        
    def test_crossover(self):
        parent_1, parent_2 = NetworkGenotype(schema), NetworkGenotype(schema)
        child = crossover(parent_1, parent_2)
        # Child must be different from both parents
        self.assertFalse(
            torch.equal(parent_1.chromosomes['conv.weight'], child.chromosomes['conv.weight']) and
            torch.equal(parent_1.chromosomes['conv.bias'], child.chromosomes['conv.bias']) and
            torch.equal(parent_1.chromosomes['fc.weight'], child.chromosomes['fc.weight']) and
            torch.equal(parent_1.chromosomes['fc.bias'], child.chromosomes['fc.bias'])
        )
        self.assertFalse(
            torch.equal(parent_2.chromosomes['conv.weight'], child.chromosomes['conv.weight']) and
            torch.equal(parent_2.chromosomes['conv.bias'], child.chromosomes['conv.bias']) and
            torch.equal(parent_2.chromosomes['fc.weight'], child.chromosomes['fc.weight']) and
            torch.equal(parent_2.chromosomes['fc.bias'], child.chromosomes['fc.bias'])
        )
        # Child must get chromosome from one of parents
        self.assertTrue(
            torch.equal(parent_1.chromosomes['conv.weight'], child.chromosomes['conv.weight']) or 
            torch.equal(parent_2.chromosomes['conv.weight'], child.chromosomes['conv.weight'])
        )
        self.assertTrue(
            torch.equal(parent_1.chromosomes['conv.bias'], child.chromosomes['conv.bias']) or 
            torch.equal(parent_2.chromosomes['conv.bias'], child.chromosomes['conv.bias'])
        )
        self.assertTrue(
            torch.equal(parent_1.chromosomes['fc.weight'], child.chromosomes['fc.weight']) or 
            torch.equal(parent_2.chromosomes['fc.weight'], child.chromosomes['fc.weight'])
        )
        self.assertTrue(
            torch.equal(parent_1.chromosomes['fc.bias'], child.chromosomes['fc.bias']) or 
            torch.equal(parent_2.chromosomes['fc.bias'], child.chromosomes['fc.bias'])
        )
        
    def test_gen_population_crossover(self):
        parents = [NetworkGenotype(schema) for i in range(5)]
        children = gen_population_crossover(parents, 10)
        self.assertEqual(len(children), 10)