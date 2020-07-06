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
    
    def test_mutate_tensor_1d(self):
        a = torch.rand((7))
        b = mutate_tensor(a, indices=[5,3,4])
        self.assertTrue(torch.equal(a[0],b[0]))
        self.assertTrue(torch.equal(a[1],b[1]))
        self.assertTrue(torch.equal(a[2],b[2]))
        self.assertFalse(torch.equal(a[3],b[3]))
        self.assertFalse(torch.equal(a[4],b[4]))
        self.assertFalse(torch.equal(a[5],b[5]))
        self.assertTrue(torch.equal(a[6],b[6]))
    
    def test_mutate_tensor_2d(self):
        a = torch.rand((8, 5))
        b = mutate_tensor(a, indices=[1,6,3,7])
        self.assertTrue(torch.equal(a[0],b[0]))
        self.assertFalse(torch.equal(a[1],b[1]))
        self.assertTrue(torch.equal(a[2],b[2]))
        self.assertFalse(torch.equal(a[3],b[3]))
        self.assertTrue(torch.equal(a[4],b[4]))
        self.assertTrue(torch.equal(a[5],b[5]))
        self.assertFalse(torch.equal(a[6],b[6]))
        self.assertFalse(torch.equal(a[7],b[7]))
    
    def test_mutate_tensor_3d(self):
        a = torch.rand((5, 3, 3))
        b = mutate_tensor(a, indices=[0,1,3])
        self.assertFalse(torch.equal(a[0],b[0]))
        self.assertFalse(torch.equal(a[1],b[1]))
        self.assertTrue(torch.equal(a[2],b[2]))
        self.assertFalse(torch.equal(a[3],b[3]))
        self.assertTrue(torch.equal(a[4],b[4]))
    
    def test_mutate_tensor_4d(self):
        a = torch.rand((7, 3, 4, 4))
        b = mutate_tensor(a, indices=[1,3,5,6])
        self.assertTrue(torch.equal(a[0],b[0]))
        self.assertFalse(torch.equal(a[1],b[1]))
        self.assertTrue(torch.equal(a[2],b[2]))
        self.assertFalse(torch.equal(a[3],b[3]))
        self.assertTrue(torch.equal(a[4],b[4]))
        self.assertFalse(torch.equal(a[5],b[5]))
        self.assertFalse(torch.equal(a[6],b[6]))
        
    # def test_gen_population_mutation(self):
    #     parents = [NetworkGenotype(schema) for i in range(5)]
    #     children = gen_population_mutation(parents, 50)
    #     self.assertEqual(len(children), 50)
        
    # def test_crossover(self):
    #     parent_1, parent_2 = NetworkGenotype(schema), NetworkGenotype(schema)
    #     child = crossover(parent_1, parent_2)
    #     # Child must be different from both parents
    #     self.assertFalse(
    #         torch.equal(parent_1.chromosomes['conv.weight'], child.chromosomes['conv.weight']) and
    #         torch.equal(parent_1.chromosomes['conv.bias'], child.chromosomes['conv.bias']) and
    #         torch.equal(parent_1.chromosomes['fc.weight'], child.chromosomes['fc.weight']) and
    #         torch.equal(parent_1.chromosomes['fc.bias'], child.chromosomes['fc.bias'])
    #     )
    #     self.assertFalse(
    #         torch.equal(parent_2.chromosomes['conv.weight'], child.chromosomes['conv.weight']) and
    #         torch.equal(parent_2.chromosomes['conv.bias'], child.chromosomes['conv.bias']) and
    #         torch.equal(parent_2.chromosomes['fc.weight'], child.chromosomes['fc.weight']) and
    #         torch.equal(parent_2.chromosomes['fc.bias'], child.chromosomes['fc.bias'])
    #     )
    #     # Child must get chromosome from one of parents
    #     self.assertTrue(
    #         torch.equal(parent_1.chromosomes['conv.weight'], child.chromosomes['conv.weight']) or 
    #         torch.equal(parent_2.chromosomes['conv.weight'], child.chromosomes['conv.weight'])
    #     )
    #     self.assertTrue(
    #         torch.equal(parent_1.chromosomes['conv.bias'], child.chromosomes['conv.bias']) or 
    #         torch.equal(parent_2.chromosomes['conv.bias'], child.chromosomes['conv.bias'])
    #     )
    #     self.assertTrue(
    #         torch.equal(parent_1.chromosomes['fc.weight'], child.chromosomes['fc.weight']) or 
    #         torch.equal(parent_2.chromosomes['fc.weight'], child.chromosomes['fc.weight'])
    #     )
    #     self.assertTrue(
    #         torch.equal(parent_1.chromosomes['fc.bias'], child.chromosomes['fc.bias']) or 
    #         torch.equal(parent_2.chromosomes['fc.bias'], child.chromosomes['fc.bias'])
    #     )
        
    # def test_gen_population_crossover(self):
    #     parents = [NetworkGenotype(schema) for i in range(5)]
    #     children = gen_population_crossover(parents, 10)
    #     self.assertEqual(len(children), 10)