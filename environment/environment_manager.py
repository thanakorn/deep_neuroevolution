import gym
import torch

class EnvironmentManager(gym.Wrapper):
    def __init__(self, env_name, device='cpu'):
        env = gym.make(env_name).unwrapped
        super().__init__(env)
        self.done = False
        self.device = device
        
    def reset(self):
        self.done = False
        state = super().reset()
        return torch.tensor(state, dtype=torch.float, device=self.device)
    
    def step(self, action):
        next_state, reward, self.done, _ = self.env.step(action)
        return (torch.tensor(next_state, dtype=torch.float, device=self.device), reward, self.done, _)
        
    def num_actions(self):
        return self.action_space.n
    
    def input_dim(self):
        return self.observation_space.shape