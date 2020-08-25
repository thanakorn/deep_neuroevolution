import torch
from collections import deque
from utilities.random_generator import NPRandomGenerator
from ray.experimental.queue import Queue

default_random_generator = NPRandomGenerator()

class ReplayMemory:
    def __init__(self, memory_size=20000, random_generator=default_random_generator, memory_ratio=1.0):
        self.memory = Queue(maxsize=memory_size)
        self.random_generator = random_generator
        self.memory_ratio = memory_ratio
        
    def add(self, data, block=False):
        if self.memory.full(): self.memory.get(True)
        if self.random_generator.rand() < self.memory_ratio: self.memory.put(data, block)
        
    def sample(self, n):
        assert n <= self.memory.size(), "Not enough replay memory"
        data = []
        while self.memory.size() > 0: data.append(self.memory.get())
        sample_idx = self.random_generator.randint(len(data), n)
        samples = [data[i] for i in sample_idx]
        return torch.stack(samples, dim=0)
        