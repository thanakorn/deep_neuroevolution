import torch
from collections import namedtuple
from typing import TypeVar, Generic, Dict

ConvSchema = namedtuple(
    'ConvSchema',
    ('in_channels', 'out_channels', 'kernel_size', 'stride')
)

LinearSchema = namedtuple(
    'LinearSchema',
    ('in_features', 'out_features')
)

ActivationSchema = namedtuple(
    'ActivationSchema',
    ('name')
)

C = TypeVar('C', ConvSchema, LinearSchema, ActivationSchema)
NetworkSchema = Dict[str, C]
