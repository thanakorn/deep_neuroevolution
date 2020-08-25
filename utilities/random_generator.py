import numpy as np

class RandomGenerator:
    def randint(self, high, n, low=0):
        raise NotImplementedError()
    
class NPRandomGenerator(RandomGenerator):
    def randint(self, high, n, low=0):
        return  np.random.choice(range(low, high), n, replace=False)
    
    def rand(self, low=0.0, high=1.0):
        return np.random.uniform(low, high)