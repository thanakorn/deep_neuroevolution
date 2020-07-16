import unittest
import torch
from genetic_algorithm.network_schema import *
from genetic_algorithm.genotype import *

class LayerGenotypeTest(unittest.TestCase):
    def test_network_genotype_create_parameters(self):
        network_schema = {
            'conv1': ConvSchema(3,16,4,4),
            'conv2': ConvSchema(16,32,5,2),
            'fc1': LinearSchema(12,8),
            'fc2': LinearSchema(8,2),
        }
        network_genotype = LayerGenotype(network_schema)
        self.assertTrue('conv1.weight' in network_genotype.genes)
        self.assertTrue('conv1.bias' in network_genotype.genes)
        self.assertTrue('conv2.weight' in network_genotype.genes)
        self.assertTrue('conv2.bias' in network_genotype.genes)
        self.assertTrue('fc1.weight' in network_genotype.genes)
        self.assertTrue('fc1.bias' in network_genotype.genes)
        self.assertTrue('fc2.weight' in network_genotype.genes)
        self.assertTrue('fc2.bias' in network_genotype.genes)
        
    def test_network_genotype_parameter_dimension(self):
        network_schema = {
            'conv1': ConvSchema(3,16,4,2),
            'fc1': LinearSchema(12,2)
        }
        network_genotype = LayerGenotype(network_schema)
        conv1_weight = network_genotype.genes['conv1.weight']
        conv1_bias = network_genotype.genes['conv1.bias']
        fc1_weight = network_genotype.genes['fc1.weight']
        fc1_bias = network_genotype.genes['fc1.bias']
        self.assertTrue(torch.Size([16, 3, 4, 4]) == conv1_weight.shape)
        self.assertTrue(torch.Size([16]) == conv1_bias.shape)
        self.assertTrue(torch.Size([2, 12]) == fc1_weight.shape)
        self.assertTrue(torch.Size([2]) == fc1_bias.shape)
        
    def test_network_genotype_clone(self):
        network_schema = {
            'conv1': ConvSchema(3,16,4,2),
            'fc1': LinearSchema(12,2)
        }
        genotype = LayerGenotype(network_schema)
        genes = genotype.genes
        clone_genotype = genotype.clone()
        cloned_genes = clone_genotype.genes
        self.assertTrue(genotype.schema == clone_genotype.schema)
        self.assertTrue(torch.equal(genes['conv1.weight'], cloned_genes['conv1.weight']))
        self.assertTrue(torch.equal(genes['conv1.bias'], cloned_genes['conv1.bias']))
        self.assertTrue(torch.equal(genes['fc1.weight'], cloned_genes['fc1.weight']))
        self.assertTrue(torch.equal(genes['fc1.bias'], cloned_genes['fc1.bias']))
        self.assertNotEqual(id(genes['fc1.bias']), id(cloned_genes['fc1.bias'])) # Ensure deepcopy
        
    def test_genoty_to_network(self):
        network_schema = {
            'conv1': ConvSchema(3,16,4,2),
            'conv2': ConvSchema(3,16,4,2),
            'relu1': ActivationSchema('ReLU'),
            'flatten': ActivationSchema('Flatten'),
            'fc1': LinearSchema(12,2),
            'relu2': ActivationSchema('ReLU'),
            'fc2': LinearSchema(12,2)
        }
        genotype = TensorGenotype(network_schema)
        network = genotype.to_network()
        self.assertTrue(isinstance(network.conv1, nn.Conv2d))
        self.assertTrue(isinstance(network.conv2, nn.Conv2d))
        self.assertTrue(isinstance(network.relu1, nn.ReLU))
        self.assertTrue(isinstance(network.relu2, nn.ReLU))
        self.assertTrue(isinstance(network.flatten, nn.Flatten))
        self.assertTrue(isinstance(network.fc1, nn.Linear))
        self.assertTrue(isinstance(network.fc2, nn.Linear))
        genotype_params = genotype.to_state_dict()
        network_params = network.state_dict()
        self.assertTrue(torch.equal(genotype_params['conv1.weight'], network_params['conv1.weight']))
        self.assertTrue(torch.equal(genotype_params['conv1.bias'], network_params['conv1.bias']))
        self.assertTrue(torch.equal(genotype_params['conv2.weight'], network_params['conv2.weight']))
        self.assertTrue(torch.equal(genotype_params['conv2.bias'], network_params['conv2.bias']))
        self.assertTrue(torch.equal(genotype_params['fc1.weight'], network_params['fc1.weight']))
        self.assertTrue(torch.equal(genotype_params['fc1.bias'], network_params['fc1.bias']))
        self.assertTrue(torch.equal(genotype_params['fc2.weight'], network_params['fc2.weight']))
        self.assertTrue(torch.equal(genotype_params['fc2.bias'], network_params['fc2.bias']))