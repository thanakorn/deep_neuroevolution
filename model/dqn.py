import torch
import torch.nn as nn
from collections import OrderedDict
from genetic_algorithm.network_schema import *
from model.genetic_network import GeneticNetwork

class DQN(GeneticNetwork):
    
    @classmethod
    def genetic_schema(cls, **model_params):
        input_channels = model_params['input_channels']
        img_size = model_params['img_size']
        num_actions = model_params['num_actions']
        conv1_output_size = int((img_size - 8) / 4) + 1
        conv2_output_size = int((conv1_output_size - 4) / 2) + 1
        schema = OrderedDict()
        schema['conv1'] = ConvSchema(input_channels, 16, 8, 4)
        schema['conv2'] = ConvSchema(16, 32, 4, 2)
        schema['fc'] = LinearSchema(32 * (conv2_output_size ** 2), 256)
        schema['output'] = LinearSchema(256, num_actions)
        return schema
        
    def forward(self, state):
        out = self.conv1(state)
        out = torch.relu(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        out = torch.relu(out)
        out = self.output(out)
        return out
        