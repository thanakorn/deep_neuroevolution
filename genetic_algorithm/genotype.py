import torch
import copy
from logging import Logger
from collections import OrderedDict 
from genetic_algorithm.chromosomes import *

class NetworkGenotype:
    def to_state_dict(self):
        raise NotImplementedError()
    
    def clone(self):
        return copy.deepcopy(self)

class LayerGenotype(NetworkGenotype):
    def __init__(self, schema: ChromosomeSchema):
        self.schema = schema
        self.genes = OrderedDict()
        for name, chromosome in schema.items():
            if isinstance(chromosome, ConvChromosome):
                in_channel, out_channel, kernel_size, _ = chromosome
                self.genes[f'{name}.weight'] = torch.rand((out_channel, in_channel, kernel_size, kernel_size))
                self.genes[f'{name}.bias'] = torch.rand(out_channel)
            elif isinstance(chromosome, LinearChromosome):
                in_feature, out_feature = chromosome
                self.genes[f'{name}.weight'] = torch.rand((out_feature, in_feature))
                self.genes[f'{name}.bias'] = torch.rand(out_feature)
                
    def to_state_dict(self):
        return self.genes            
    
class TensorGenotype(NetworkGenotype):
    def __init__(self, schema: ChromosomeSchema):
        self.schema = schema
        self.genes = []
        for name, chromosome in schema.items():
            if isinstance(chromosome, ConvChromosome):
                in_channel, out_channel, kernel_size, _ = chromosome
                for _ in range(out_channel): self.genes.append(torch.rand(in_channel, kernel_size, kernel_size)) # conv kernels
                self.genes.append(torch.rand(out_channel)) # bias
            if isinstance(chromosome, LinearChromosome):
                in_feature, out_feature = chromosome
                for _ in range(out_feature): self.genes.append(torch.rand(in_feature)) # weights
                self.genes.append(torch.rand(out_feature)) # bias
                
    def to_state_dict(self):
        state_dict = {}
        start = 0
        for name, chromosome in self.schema.items():
            if isinstance(chromosome, ConvChromosome):
                _, out_channel, _, _ = chromosome
                state_dict[f'{name}.weight'] = torch.stack(self.genes[start:start + out_channel], dim=0)
                state_dict[f'{name}.bias'] = self.genes[start + out_channel]
                start += out_channel + 1
            if isinstance(chromosome, LinearChromosome):
                _, out_feature = chromosome
                state_dict[f'{name}.weight'] = torch.stack(self.genes[start:start + out_feature], dim=0)
                state_dict[f'{name}.bias'] = self.genes[start + out_feature]
                start += out_feature + 1
        return state_dict
                