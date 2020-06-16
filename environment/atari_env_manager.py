import gym
import torch
import numpy as np
import cv2

from collections import deque
from environment.environment_manager import EnvironmentManager

IMAGE_SIZE  = (84, 84)

class AtariEnvManager(EnvironmentManager):
    def __init__(self, env_name, frame_stack_size=4):
        super().__init__(gym.make(env_name).unwrapped)
        self.frames = deque([], maxlen=frame_stack_size)
        
    def reset(self):
        super().reset()
        screen = self.get_processed_screen()
        for _ in range(self.frames.maxlen - 1):
            self.frames.append(np.zeros_like(screen))
        self.frames.append(screen)
        return self.state()
        
    """
    State is a stack of tensors representing last k images
    """
    def state(self):
        return torch.from_numpy(np.stack(self.frames))
    
    """
    Take action and return reward, next_state, and done
    """
    def step(self, action):
        _, reward, self.done, _ = self.env.step(action)
        screen = self.get_processed_screen()
        self.frames.append(screen)
        next_state = self.state()
        return (torch.tensor([reward]), next_state, self.done)
        
    def get_processed_screen(self):
        screen = cv2.cvtColor(self.get_raw_screen(), cv2.COLOR_RGB2GRAY)
        screen = cv2.resize(screen, IMAGE_SIZE)
        screen = np.ascontiguousarray(screen)
        return screen
    
    def get_raw_screen(self):
        return self.render('rgb_array')