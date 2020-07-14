import gym
import torch
import numpy as np
import cv2

from collections import deque
from environment.environment_manager import EnvironmentManager

IMAGE_SIZE  = (84, 84)

class FrameStackEnvManager(EnvironmentManager):
    def __init__(self, env_name, device='cpu', frame_stack_size=4):
        super().__init__(env_name, device)
        self.frames = deque([], maxlen=frame_stack_size)
        
    def reset(self):
        self.env.reset()
        screen = self.get_raw_screen()
        screen = self.processed_screen(screen)
        for _ in range(self.frames.maxlen): self.frames.append(screen)
        return self.state()

    def state(self):
        return torch.from_numpy(np.stack(self.frames)).float()
    
    def step(self, action):
        screen, reward, self.done, _ = self.env.step(action)
        screen = self.processed_screen(screen)
        self.frames.append(screen)
        return (self.state(), torch.tensor([reward]), self.done, _)
        
    def processed_screen(self, screen):
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        screen = cv2.resize(screen, IMAGE_SIZE)
        screen = np.ascontiguousarray(screen)
        return screen
    
    def get_raw_screen(self):
        return self.env.render('rgb_array')