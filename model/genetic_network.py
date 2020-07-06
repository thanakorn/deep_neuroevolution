import torch
import torch.nn as nn
from genetic_algorithm.genotype import NetworkGenotype
from genetic_algorithm.chromosomes import ChromosomeSchema, ConvChromosome, LinearChromosome

class GeneticNetwork(nn.Module):     
    @classmethod
    def genetic_schema(cls) -> ChromosomeSchema:
        raise NotImplementedError()
    
    def __init__(self, genotype: NetworkGenotype):
        super().__init__()
        for name, chromosome in genotype.schema.items():
            if isinstance(chromosome, ConvChromosome):
                in_channels, out_channels, kernel_size, stride = chromosome
                module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
            elif isinstance(chromosome, LinearChromosome):
                in_features, out_features = chromosome
                module = nn.Linear(in_features, out_features)    
            self.add_module(name, module)
            
        self.set_weigths(genotype)
        
    def set_weigths(self, genotype: NetworkGenotype):
        self.load_state_dict(genotype.chromosomes)