import unittest
import torch
import cv2
import numpy as np
import ray
from environment.framestack_env_manager import FrameStackEnvManager, DEFAULT_IMAGE_SIZE
from memory.replay_memory import ReplayMemory

img_size = (64, 64)

def preprocess(screen):
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    screen = cv2.resize(screen, img_size, interpolation=cv2.INTER_NEAREST)
    screen = np.ascontiguousarray(screen)
    return screen

class FrameStackEnvManagerTest(unittest.TestCase):
    def test_env_manager_init(self):
        stack_size = 3
        env = FrameStackEnvManager('Breakout-v0', preprocess=preprocess, frame_stack_size=stack_size)
        self.assertEqual(0, len(env.frames))
        env.reset()
        self.assertEqual(env.state()[0].shape, torch.Size(img_size))
        self.assertEqual(env.state().shape[0], stack_size)
        
    def test_env_manager_reset(self):
        env = FrameStackEnvManager('Pong-v0', preprocess=preprocess, frame_stack_size=4)
        state = env.reset()
        screen = torch.tensor(preprocess(env.render('rgb_array'))).float()
        # All frames in the stack is similar
        self.assertTrue(torch.equal(state[0], screen))
        self.assertTrue(torch.equal(state[1], screen))
        self.assertTrue(torch.equal(state[2], screen))
        self.assertTrue(torch.equal(state[3], screen))
        
    def test_env_manager_state(self):
        frame_stack_size = 5
        env = FrameStackEnvManager('Breakout-v0', preprocess=preprocess, frame_stack_size=frame_stack_size)
        env.reset()
        state = env.state()
        self.assertEqual(frame_stack_size, state.shape[0])
        
    def test_env_manager_step(self):
        env = FrameStackEnvManager('Pong-v0', preprocess=preprocess)
        start_state = env.reset()
        next_state, _, _, _ = env.step(env.action_space.sample())
        new_frame = env.env.render('rgb_array')
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        new_frame = cv2.resize(new_frame, img_size, interpolation=cv2.INTER_NEAREST)
        new_frame = torch.tensor(new_frame).float()
        self.assertTrue(torch.equal(next_state[0], start_state[1]))
        self.assertTrue(torch.equal(next_state[1], start_state[2]))
        self.assertTrue(torch.equal(next_state[2], start_state[3]))
        self.assertTrue(torch.equal(next_state[3], new_frame))
        
        next_state2, _, _, _ = env.step(env.action_space.sample())
        new_frame = env.env.render('rgb_array')
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        new_frame = cv2.resize(new_frame, img_size, interpolation=cv2.INTER_NEAREST)
        new_frame = torch.tensor(new_frame).float()
        self.assertTrue(torch.equal(next_state2[0], next_state[1]))
        self.assertTrue(torch.equal(next_state2[1], next_state[2]))
        self.assertTrue(torch.equal(next_state2[2], next_state[3]))
        self.assertTrue(torch.equal(next_state2[3], new_frame))
        
        next_state3, _, _, _ = env.step(env.action_space.sample())
        next_state4, _, _, _ = env.step(env.action_space.sample())
        new_frame = env.env.render('rgb_array')
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        new_frame = cv2.resize(new_frame, img_size, interpolation=cv2.INTER_NEAREST)
        new_frame = torch.tensor(new_frame).float()
        self.assertTrue(torch.equal(next_state4[0], next_state[3]))
        self.assertTrue(torch.equal(next_state4[1], next_state2[3]))
        self.assertTrue(torch.equal(next_state4[2], next_state3[3]))
        self.assertTrue(torch.equal(next_state4[3], new_frame))
        
    def test_env_manager_replay_memory(self):
        ray.init()
        memory = ReplayMemory()
        env = FrameStackEnvManager('Pong-v0', preprocess=preprocess, replay_memory=memory)
        env.reset()
        for i in range(10): env.step(env.action_space.sample())
        self.assertEqual(len(memory.memory), 10)
        ray.shutdown()
            