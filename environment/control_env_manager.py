import gym
import torch
import numpy as np

from environment.environment_manager import EnvironmentManager

class ControlEnvManager(EnvironmentManager):
    def __init__(self, env_name):
        super().__init__(gym.make(env_name).unwrapped)
        
    def reset(self):
        state = super().reset()
        return torch.tensor(state, dtype=torch.float)
    
    def step(self, action):
        next_state, reward, self.done, _ = self.env.step(action)
        return (torch.tensor(next_state, dtype=torch.float), torch.tensor([reward]), self.done, _)