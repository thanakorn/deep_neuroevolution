import torch
import torch.nn as nn
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.network_schema import *

class GeneticNetwork(nn.Module):     
    @classmethod
    def genetic_schema(cls, **model_params) -> NetworkSchema:
        raise NotImplementedError()
    
    def __init__(self, genotype: NetworkGenotype):
        super().__init__()
        for name, module_schema in genotype.schema.items():
            if isinstance(module_schema, ConvSchema):
                in_channels, out_channels, kernel_size, stride = module_schema
                module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
            elif isinstance(module_schema, ConvSchema):
                in_features, out_features = module_schema
                module = nn.Linear(in_features, out_features)    
            self.add_module(name, module)
            
        self.set_weigths(genotype)
        
    def set_weigths(self, genotype: NetworkGenotype):
        self.load_state_dict(genotype.to_state_dict())