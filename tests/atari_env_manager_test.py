import unittest
import torch
import cv2
from environment.atari_env_manager import AtariEnvManager, IMAGE_SIZE

class AtariEnvManagerTest(unittest.TestCase):
    def test_env_manager_init(self):
        env = AtariEnvManager('Breakout-v0')
        self.assertEqual(0, len(env.frames))
        
    def test_env_manager_reset(self):
        env = AtariEnvManager('Pong-v0', frame_stack_size=4)
        state = env.reset()
        # All frames in the stack is similar
        self.assertTrue(torch.equal(state[0], state[1]))
        self.assertTrue(torch.equal(state[1], state[2]))
        self.assertTrue(torch.equal(state[2], state[3]))
        
    def test_env_manager_processed_screen(self):
        env = AtariEnvManager('Pong-v0')
        raw_screen = env.env.render('rgb_array')
        processed_screen = env.processed_screen(raw_screen)
        self.assertEqual(2, processed_screen.ndim) # Convert RGB to Gray
        
    def test_env_manager_state(self):
        frame_stack_size = 5
        env = AtariEnvManager('Breakout-v0', frame_stack_size)
        env.reset()
        state = env.state()
        self.assertEqual(frame_stack_size, state.shape[0])
        
    def test_env_manager_step(self):
        env = AtariEnvManager('Pong-v0')
        start_state = env.reset()
        next_state, _, _, _ = env.step(env.action_space.sample())
        new_frame = env.env.render('rgb_array')
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        new_frame = cv2.resize(new_frame, IMAGE_SIZE)
        new_frame = torch.tensor(new_frame).float()
        self.assertTrue(torch.equal(next_state[0], start_state[1]))
        self.assertTrue(torch.equal(next_state[1], start_state[2]))
        self.assertTrue(torch.equal(next_state[2], start_state[3]))
        self.assertTrue(torch.equal(next_state[3], new_frame))
        
        next_state2, _, _, _ = env.step(env.action_space.sample())
        new_frame = env.env.render('rgb_array')
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        new_frame = cv2.resize(new_frame, IMAGE_SIZE)
        new_frame = torch.tensor(new_frame).float()
        self.assertTrue(torch.equal(next_state2[0], next_state[1]))
        self.assertTrue(torch.equal(next_state2[1], next_state[2]))
        self.assertTrue(torch.equal(next_state2[2], next_state[3]))
        self.assertTrue(torch.equal(next_state2[3], new_frame))
        
        next_state3, _, _, _ = env.step(env.action_space.sample())
        next_state4, _, _, _ = env.step(env.action_space.sample())
        new_frame = env.env.render('rgb_array')
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        new_frame = cv2.resize(new_frame, IMAGE_SIZE)
        new_frame = torch.tensor(new_frame).float()
        self.assertTrue(torch.equal(next_state4[0], next_state[3]))
        self.assertTrue(torch.equal(next_state4[1], next_state2[3]))
        self.assertTrue(torch.equal(next_state4[2], next_state3[3]))
        self.assertTrue(torch.equal(next_state4[3], new_frame))