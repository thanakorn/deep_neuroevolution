import numpy as np

class RandomGenerator:
    def randint(self, high, n, low=0):
        raise NotImplementedError()
    
class NPRandomGenerator(RandomGenerator):
    def randint(self, high, n, low=0):
        return  np.random.randint(low, high, n)
        