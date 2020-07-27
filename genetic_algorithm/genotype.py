import torch
import torch.nn as nn
import copy
from logging import Logger
from collections import OrderedDict 
from genetic_algorithm.network_schema import *

class NetworkGenotype:
    def __init__(self, schema: NetworkSchema, init_func=None):
        self.schema = schema
        self.network = self.create_init_network()
        
    def create_init_network(self):
        network = nn.Sequential()
        for name, module_schema in self.schema.items():
            if isinstance(module_schema, ConvSchema):
                in_channels, out_channels, kernel_size, stride = module_schema
                module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
            elif isinstance(module_schema, LinearSchema):
                in_features, out_features = module_schema
                module = nn.Linear(in_features=in_features, out_features=out_features)
            elif isinstance(module_schema, ActivationSchema):
                module = self.get_activation_module(module_schema[0])
            network.add_module(name, module)
        return network
    
    def to_state_dict(self):
        raise NotImplementedError()
    
    def clone(self):
        return copy.deepcopy(self)
    
    def to_network(self):
        state_dict = self.to_state_dict()
        self.network.load_state_dict(state_dict)
        return self.network
        
    def get_activation_module(self, name):
        if name == 'ReLU': return nn.ReLU()
        if name == 'Flatten': return nn.Flatten()

class LayerGenotype(NetworkGenotype):
    def __init__(self, schema: NetworkSchema, init_func=None):
        super().__init__(schema, init_func)
        self.genes = OrderedDict()
        for name, module_schema in schema.items():
            if isinstance(module_schema, ConvSchema):
                in_channel, out_channel, kernel_size, _ = module_schema
                weight = torch.rand((out_channel, in_channel, kernel_size, kernel_size))
                bias = torch.zeros(out_channel)
            elif isinstance(module_schema, LinearSchema):
                in_feature, out_feature = module_schema
                weight = torch.rand((out_feature, in_feature))
                bias = torch.zeros(out_feature)
                
            self.genes[f'{name}.weight'] = init_func(weight) if init_func is not None else weight
            self.genes[f'{name}.bias'] = bias
                
    def to_state_dict(self):
        return self.genes            
    
class TensorGenotype(NetworkGenotype):
    def __init__(self, schema: NetworkSchema, init_func=None):
        super().__init__(schema, init_func)
        self.genes = []
        for name, module_schema in schema.items():
            if isinstance(module_schema, ConvSchema):
                in_channel, out_channel, kernel_size, _ = module_schema
                for _ in range(out_channel):
                    w = torch.rand(in_channel, kernel_size, kernel_size)
                    w = init_func(w) if init_func is not None else w
                    self.genes.append(w) # conv kernels
                self.genes.append(torch.zeros(out_channel)) # bias
            if isinstance(module_schema, LinearSchema):
                in_feature, out_feature = module_schema
                for _ in range(out_feature):
                    w = torch.rand(in_feature, 1)
                    w = init_func(w) if init_func is not None else w
                    self.genes.append(w.squeeze()) # weights
                self.genes.append(torch.zeros(out_feature)) # bias
                
    def to_state_dict(self):
        state_dict = {}
        start = 0
        for name, module_schema in self.schema.items():
            if isinstance(module_schema, ConvSchema):
                _, out_channel, _, _ = module_schema
                state_dict[f'{name}.weight'] = torch.stack(self.genes[start:start + out_channel], dim=0)
                state_dict[f'{name}.bias'] = self.genes[start + out_channel]
                start += out_channel + 1
            if isinstance(module_schema, LinearSchema):
                _, out_feature = module_schema
                state_dict[f'{name}.weight'] = torch.stack(self.genes[start:start + out_feature], dim=0)
                state_dict[f'{name}.bias'] = self.genes[start + out_feature]
                start += out_feature + 1
        return state_dict
                