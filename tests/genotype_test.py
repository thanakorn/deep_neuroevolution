import unittest
import torch
from genetic_algorithm.chromosomes import ConvChromosome, LinearChromosome
from genetic_algorithm.genotype import NetworkGenotype

class NetworkGenotypeTest(unittest.TestCase):
    def test_network_genotype_create_parameters(self):
        network_schema = {
            'conv1': ConvChromosome(3,16,4,4),
            'conv2': ConvChromosome(16,32,5,2),
            'fc1': LinearChromosome(12,8),
            'fc2': LinearChromosome(8,2),
        }
        network_genotype = NetworkGenotype(network_schema)
        self.assertTrue('conv1.weight' in network_genotype.chromosomes)
        self.assertTrue('conv1.bias' in network_genotype.chromosomes)
        self.assertTrue('conv2.weight' in network_genotype.chromosomes)
        self.assertTrue('conv2.bias' in network_genotype.chromosomes)
        self.assertTrue('fc1.weight' in network_genotype.chromosomes)
        self.assertTrue('fc1.bias' in network_genotype.chromosomes)
        self.assertTrue('fc2.weight' in network_genotype.chromosomes)
        self.assertTrue('fc2.bias' in network_genotype.chromosomes)
        
    def test_network_genotype_parameter_dimension(self):
        network_schema = {
            'conv1': ConvChromosome(3,16,4,2),
            'fc1': LinearChromosome(12,2)
        }
        network_genotype = NetworkGenotype(network_schema)
        conv1_weight = network_genotype.chromosomes['conv1.weight']
        conv1_bias = network_genotype.chromosomes['conv1.bias']
        fc1_weight = network_genotype.chromosomes['fc1.weight']
        fc1_bias = network_genotype.chromosomes['fc1.bias']
        self.assertTrue(torch.Size([16, 3, 4, 4]) == conv1_weight.shape)
        self.assertTrue(torch.Size([16]) == conv1_bias.shape)
        self.assertTrue(torch.Size([2, 12]) == fc1_weight.shape)
        self.assertTrue(torch.Size([2]) == fc1_bias.shape)
        
    def test_network_genotype_clone(self):
        network_schema = {
            'conv1': ConvChromosome(3,16,4,2),
            'fc1': LinearChromosome(12,2)
        }
        genotype = NetworkGenotype(network_schema)
        chromosomes = genotype.chromosomes
        clone_genotype = genotype.clone()
        cloned_chromosomes = clone_genotype.chromosomes
        self.assertTrue(genotype.schema == clone_genotype.schema)
        self.assertTrue(torch.equal(chromosomes['conv1.weight'], cloned_chromosomes['conv1.weight']))
        self.assertTrue(torch.equal(chromosomes['conv1.bias'], cloned_chromosomes['conv1.bias']))
        self.assertTrue(torch.equal(chromosomes['fc1.weight'], cloned_chromosomes['fc1.weight']))
        self.assertTrue(torch.equal(chromosomes['fc1.bias'], cloned_chromosomes['fc1.bias']))
        clone_genotype.chromosomes['fc1.bias'][0] = 0.
        self.assertFalse(torch.equal(chromosomes['fc1.bias'], cloned_chromosomes['fc1.bias'])) # Ensure deepcopy