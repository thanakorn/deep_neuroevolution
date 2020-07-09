import gym
import torch
import numpy as np
import cv2

from collections import deque
from environment.environment_manager import EnvironmentManager

IMAGE_SIZE = (60, 40)

class ClassicControlEnvManager(EnvironmentManager):
    def __init__(self, env_name, frame_stack_size=4):
        super().__init__(gym.make(env_name).unwrapped)
        
    def reset(self):
        super().reset()
        screen = self.env.render('rgb_array')
        screen = self.processed_screen(screen)
        return self.state()
        
    """Return a stack of tensors representing last k frames"""
    def state(self):
        screen = self.env.render('rgb_array')
        screen = self.processed_screen(screen)
        return torch.from_numpy(screen).float()
    
    def step(self, action):
        _, reward, self.done, _ = self.env.step(action)
        return (self.state(), torch.tensor([reward]), self.done, _)
        
    def processed_screen(self, screen):
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        screen = cv2.resize(screen, IMAGE_SIZE)
        h, w = screen.shape
        screen = np.ascontiguousarray(screen)
        return screen
    
    def get_raw_screen(self):
        return self.env.render('rgb_array')