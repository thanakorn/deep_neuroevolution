import unittest
import torch
import torch.nn as nn
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.chromosomes import *
from model.genetic_network import GeneticNetwork
from model.simple_cnn import SimpleCNN

class TestModel(GeneticNetwork):
    @classmethod
    def genetic_schema(cls):
        schema = {
            'conv': ConvChromosome(3, 16, 3, 4),
            'fc': LinearChromosome(128, 3)
        }
        return schema

class ModelTest(unittest.TestCase):
    
    def test_create_network_from_schema(self):
        genotype = NetworkGenotype(TestModel.genetic_schema())
        model = TestModel(genotype)
        self.assertTrue(isinstance(getattr(model,'conv'), nn.Conv2d))
        self.assertTrue(isinstance(getattr(model,'fc'), nn.Linear))
    
    def test_create_network_weight(self):
        genotype = NetworkGenotype(TestModel.genetic_schema())
        model = TestModel(genotype)
        self.assertTrue(torch.equal(model.conv.weight, genotype.chromosomes['conv.weight']))
        self.assertTrue(torch.equal(model.conv.bias, genotype.chromosomes['conv.bias']))
        self.assertTrue(torch.equal(model.fc.weight, genotype.chromosomes['fc.weight']))
        self.assertTrue(torch.equal(model.fc.bias, genotype.chromosomes['fc.bias']))