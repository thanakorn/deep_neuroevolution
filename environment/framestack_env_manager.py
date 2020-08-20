import gym
import torch
import numpy as np
import cv2

from collections import deque
from environment.environment_manager import EnvironmentManager

DEFAULT_IMAGE_SIZE  = (84, 84)

class FrameStackEnvManager(EnvironmentManager):
    def __init__(self, env_name, preprocess, device='cpu', frame_stack_size=4, replay_memory=None):
        super().__init__(env_name, device, replay_memory)
        self.frames = deque([], maxlen=frame_stack_size)
        self.preprocess = preprocess
        
    def reset(self):
        screen = self.env.reset()
        screen = self.preprocess(screen)
        for _ in range(self.frames.maxlen): self.frames.append(screen)
        return self.state()

    def state(self):
        return torch.from_numpy(np.stack(self.frames)).float().to(self.device)
    
    def step(self, action):
        screen, reward, self.done, _ = self.env.step(action)
        screen = self.preprocess(screen)
        self.frames.append(screen)
        next_state = self.state()
        if self.replay_memory is not None: self.replay_memory.add(next_state)
        return (next_state, torch.tensor([reward]), self.done, _)
    
    def get_raw_screen(self):
        raw_screen = self.env.render('rgb_array')
        return raw_screen