import torch
from collections import deque
from utilities.random_generator import NPRandomGenerator

random_generator = NPRandomGenerator()

class ReplayMemory:
    def __init__(self, memory_size=20000, random_generator=random_generator):
        self.memory = deque([], memory_size)
        self.random_generator = random_generator
        
    def add(self, data):
        self.memory.append(data)
        
    def sample(self, n):
        assert n <= len(self.memory), "Not enough replay memory"
        sample_idx = random_generator.randint(len(self.memory), n)
        samples = [self.memory[i] for i in sample_idx]
        return torch.stack(samples, dim=0)
        