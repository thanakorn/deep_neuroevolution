import unittest
import torch
from memory.replay_memory import ReplayMemory
from utilities.random_generator import NPRandomGenerator

class ReplayMemoryTest(unittest.TestCase):
    def test_replay_memory_init(self):
        replay_memory = ReplayMemory(1000)
        self.assertEqual(replay_memory.memory.maxlen, 1000)
        
    def test_replay_memory_add(self):
        replay_memory = ReplayMemory()
        data1 = torch.randn((3,3))
        data2 = torch.randn((3,3))
        data3 = torch.randn((3,3))
        
        replay_memory.add(data1)
        replay_memory.add(data2)
        replay_memory.add(data3)
        
        self.assertTrue(torch.equal(data1, replay_memory.memory[0]))
        self.assertTrue(torch.equal(data2, replay_memory.memory[1]))
        self.assertTrue(torch.equal(data3, replay_memory.memory[2]))
        
    def test_replay_memory_limit(self):
        replay_memory = ReplayMemory(memory_size=3)
        data1 = torch.randn((3,3))
        data2 = torch.randn((3,3))
        data3 = torch.randn((3,3))
        data4 = torch.randn((3,3))
        data5 = torch.randn((3,3))
        
        replay_memory.add(data1)
        replay_memory.add(data2)
        replay_memory.add(data3)
        replay_memory.add(data4)
        
        self.assertTrue(torch.equal(data2, replay_memory.memory[0]))
        self.assertTrue(torch.equal(data3, replay_memory.memory[1]))
        self.assertTrue(torch.equal(data4, replay_memory.memory[2]))
        
        replay_memory.add(data5)
        
        self.assertTrue(torch.equal(data3, replay_memory.memory[0]))
        self.assertTrue(torch.equal(data4, replay_memory.memory[1]))
        self.assertTrue(torch.equal(data5, replay_memory.memory[2]))
        
    def test_replay_memory_sample(self):
        replay_memory = ReplayMemory(memory_size=10)
        for i in range(10): replay_memory.add(torch.randn((3,3)))
        samples = replay_memory.sample(8)
        self.assertEqual(samples.shape[0], 8)
        for i in range(8):
            for j in range(i+1, 8):
                self.assertFalse(torch.equal(samples[i], samples[j])) # No duplicate memory
        
        
        
        
        
        
        