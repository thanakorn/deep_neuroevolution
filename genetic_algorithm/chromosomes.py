import torch
from collections import namedtuple
from typing import TypeVar, Generic, Dict

ConvChromosome = namedtuple(
    'ConvChromosome',
    ('in_channels', 'out_channels', 'kernel_size', 'stride')
)

LinearChromosome = namedtuple(
    'LinearChromosome',
    ('in_features', 'out_features')
)

C = TypeVar('C', ConvChromosome, LinearChromosome)
ChromosomeSchema = Dict[str, C]
