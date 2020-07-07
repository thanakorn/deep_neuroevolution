import unittest
import torch
from genetic_algorithm.chromosomes import ConvChromosome, LinearChromosome
from genetic_algorithm.genotype import NetworkGenotype, NetworkGranularGenotype

schema = {
            'conv1': ConvChromosome(4,16,8,8),
            'conv2': ConvChromosome(16,32,4,4),
            'fc': LinearChromosome(2592, 256),
            'output': LinearChromosome(256,6),
        }

class GranularGenotypeTest(unittest.TestCase):
    def test_generate_genes(self):
        genotype = NetworkGranularGenotype(schema)
        genes = genotype.genes
        self.assertTrue(len(genes) == 314)
        # conv1
        for i in range(16): self.assertTrue(genes[i].shape == torch.Size([4, 8, 8]))
        self.assertTrue(genes[16].shape == torch.Size([16]))
        # conv2
        for i in range(17, 49): self.assertTrue(genes[i].shape == torch.Size([16, 4, 4]))
        self.assertTrue(genes[49].shape == torch.Size([32]))
        # fc
        for i in range(50,306): self.assertTrue(genes[i].shape == torch.Size([2592]))
        self.assertTrue(genes[306].shape == torch.Size([256]))
        # output
        for i in range(307, 313): self.assertTrue(genes[i].shape == torch.Size([256]))
        self.assertTrue(genes[313].shape == torch.Size([6]))
        
    def test_state_dict_keys(self):
        genotype = NetworkGranularGenotype(schema)
        state_dict = genotype.to_state_dict()
        self.assertTrue('conv1.weight' in state_dict)
        self.assertTrue('conv1.bias' in state_dict)
        self.assertTrue('conv2.weight' in state_dict)
        self.assertTrue('conv2.bias' in state_dict)
        self.assertTrue('fc.weight' in state_dict)
        self.assertTrue('fc.bias' in state_dict)
        self.assertTrue('output.weight' in state_dict)
        self.assertTrue('output.bias' in state_dict)
        
    def test_state_dict_values(self):
        state_dict = NetworkGranularGenotype(schema).to_state_dict()
        self.assertEqual(len(state_dict.keys()), 8)
        self.assertTrue(torch.Size([16,4,8,8]) == state_dict['conv1.weight'].shape)
        self.assertTrue(torch.Size([16]) == state_dict['conv1.bias'].shape)
        self.assertTrue(torch.Size([32,16,4,4]) == state_dict['conv2.weight'].shape)
        self.assertTrue(torch.Size([32]) == state_dict['conv2.bias'].shape)
        self.assertTrue(torch.Size([256,2592]) == state_dict['fc.weight'].shape)
        self.assertTrue(torch.Size([256]) == state_dict['fc.bias'].shape)
        self.assertTrue(torch.Size([6,256]) == state_dict['output.weight'].shape)
        self.assertTrue(torch.Size([6]) == state_dict['output.bias'].shape)
        
        