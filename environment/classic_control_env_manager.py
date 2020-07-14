import gym
import torch
import numpy as np
import cv2
import torchvision.transforms as T

from collections import deque
from environment.environment_manager import EnvironmentManager

process_screen = T.Compose([
                    T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize((100, 150)),
                    T.CenterCrop(64),
                    T.ToTensor()
                ])

class ClassicControlEnvManager(EnvironmentManager):
    def __init__(self, env_name):
        super().__init__(gym.make(env_name).unwrapped)
        
    def reset(self):
        super().reset()
        return self.state()
        
    """Return a stack of tensors representing last k frames"""
    def state(self):
        screen = self.env.render('rgb_array')
        return self.processed_screen(screen)
    
    def step(self, action):
        _, reward, self.done, _ = self.env.step(action)
        return (self.state(), torch.tensor([reward]), self.done, _)
        
    def processed_screen(self, screen):
        processed_screen = process_screen(screen).squeeze(0)
        return processed_screen
    
    def get_raw_screen(self):
        return self.env.render('rgb_array')