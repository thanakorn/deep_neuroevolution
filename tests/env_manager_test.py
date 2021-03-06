import unittest
import torch

from environment.environment_manager import EnvironmentManager

class EnvironmentManagerTest(unittest.TestCase):
    def test_env_manager_init(self):
        env = EnvironmentManager('CartPole-v0', 'cpu')
        self.assertEqual(env.done, False)
        self.assertEqual(env.device, 'cpu')
        
    def test_env_manager_reset(self):
        env = EnvironmentManager('CartPole-v0', 'cpu')
        state = env.reset()
        self.assertTrue(isinstance(state, torch.Tensor))
        self.assertTrue(state.device, 'cpu')
        self.assertEqual(state.dtype, torch.float)
        
    def test_env_manager_step(self):
        env = EnvironmentManager('AirRaid-ram-v0', 'cpu')
        env.reset()
        next_state, reward, done, info = env.step(env.action_space.sample())
        self.assertTrue(isinstance(next_state, torch.Tensor))
        self.assertEqual(next_state.dtype, torch.float)
        self.assertTrue(isinstance(reward, float))
        self.assertTrue(isinstance(done, bool))