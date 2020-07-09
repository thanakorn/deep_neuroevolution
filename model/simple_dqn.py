import torch
import torch.nn as nn
from collections import OrderedDict
from genetic_algorithm.chromosomes import *
from model.genetic_network import GeneticNetwork

class SimpleDQN(GeneticNetwork):
    
    @classmethod
    def genetic_schema(cls, **model_params):
        input_channels = model_params['input_channels']
        img_size = model_params['img_size']
        num_actions = model_params['num_actions']
        conv_output_size = int((img_size - 5) / 2) + 1
        schema = OrderedDict()
        schema['conv'] = ConvChromosome(input_channels, 8, 5, 2)
        schema['fc'] = LinearChromosome(8 * (conv_output_size ** 2), 32)
        schema['output'] = LinearChromosome(32, num_actions)
        return schema
    
    def forward(self, state):
        out = self.conv(state)
        out = torch.relu(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        out = torch.relu(out)
        out = self.output(out)
        return out