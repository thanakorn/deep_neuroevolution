import gym
import torch
import numpy as np

class EnvironmentManager(gym.Wrapper):
    def __init__(self, env_name, device='cpu', replay_memory=None):
        super().__init__(gym.make(env_name))
        self.done = False
        self.device = device
        self.replay_memory = replay_memory
        
    def reset(self):
        self.done = False
        state = super().reset()
        return torch.tensor(state, dtype=torch.float, device=self.device)
    
    def step(self, action):
        next_state, reward, self.done, _ = self.env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        if self.replay_memory is not None: self.replay_memory.add(next_state)
        return (next_state, reward, self.done, _)
        
    def num_actions(self):
        return self.action_space.n
    
    def input_dim(self):
        return self.observation_space.shape