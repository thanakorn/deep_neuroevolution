import gym
import numpy as np
import torch
import torchvision.transforms as T

class EnvironmentManager(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.done = False
        
    def reset(self):
        self.done = False
        return super().reset()
        
    def num_actions(self):
        return self.action_space.n
    
    def state(self):
        raise NotImplementedError()