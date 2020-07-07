import unittest
import torch
import torch.nn as nn
from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.chromosomes import *
from model.genetic_network import GeneticNetwork

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
        genotype = TensorGenotype(TestModel.genetic_schema())
        model = TestModel(genotype)
        self.assertTrue(isinstance(getattr(model,'conv'), nn.Conv2d))
        self.assertTrue(isinstance(getattr(model,'fc'), nn.Linear))
    
    def test_create_network_weight(self):
        genotype = TensorGenotype(TestModel.genetic_schema())
        model = TestModel(genotype)
        state_dict = genotype.to_state_dict()
        self.assertTrue(torch.equal(model.conv.weight, state_dict['conv.weight']))
        self.assertTrue(torch.equal(model.conv.bias, state_dict['conv.bias']))
        self.assertTrue(torch.equal(model.fc.weight, state_dict['fc.weight']))
        self.assertTrue(torch.equal(model.fc.bias, state_dict['fc.bias']))