import unittest
import torch
import ray

from genetic_algorithm.diversity_evaluator import TrajectoryDiversityEvaluator
from memory.replay_memory import ReplayMemory
from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.network_schema import *
from unittest.mock import Mock
from utilities.random_generator import RandomGenerator

schema = {
            'conv1': ConvSchema(1,4,3,2),
            'flatten': ActivationSchema('Flatten'),
            'output': LinearSchema(36,3)
         }
global memory
memory = None
populations = [TensorGenotype(schema) for i in range(4)]
    
class DiversityEvaluatorTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        global memory
        ray.init()
        memory = ReplayMemory(memory_size=32)
        for i in range(32): memory.add(torch.rand(8, 8).unsqueeze(0))
      
    @classmethod  
    def tearDownClass(cls):
        ray.shutdown()
    
    def test_diversity_evaluator_init(self):
        global memory
        evaluator = TrajectoryDiversityEvaluator(memory, 8)
        self.assertEqual(evaluator.replay_memory, memory)
        self.assertEqual(evaluator.num_samples, 8)
        
    def test_diversity_evaluator_compute_trajectory(self):
        global memory
        evaluator = TrajectoryDiversityEvaluator(memory, 8)
        samples = memory.sample(5)
        trajectories = evaluator.compute_trajectory([p.to_network() for p in populations], samples)
        
        for p_i in range(len(populations)):
            p = populations[p_i].to_network()
            q_values = p(samples)
            t = trajectories[p_i]
            for s_i in range(samples.shape[0]):
                self.assertTrue(torch.equal(q_values[s_i].argmax(), t[s_i]))
                
    def test_diversity_evaluator_eval_diversity(self):
        random_generator = RandomGenerator()
        mem_idx = [0, 3, 5, 11, 1, 18, 20, 2]
        random_generator.randint = Mock(return_value=mem_idx)
        replay_memory = ReplayMemory(32, random_generator)
        replay_memory_2 = ReplayMemory(32, random_generator)
        for i in range(32):
            s = torch.rand(8, 8).unsqueeze(0)
            replay_memory.add(s)
            replay_memory_2.add(s)
        
        evaluator = TrajectoryDiversityEvaluator(replay_memory, 8)
        diversity_scores = evaluator.eval_diversity(populations)
        self.assertEqual(len(diversity_scores), len(populations))
        
        trajectories = evaluator.compute_trajectory([p.to_network() for p in populations], replay_memory_2.sample(8))
        for i in range(trajectories.shape[0]):
            score = 0
            for j in range(trajectories.shape[0]):
                if i == j: continue
                score += (trajectories[i] != trajectories[j]).sum()
            self.assertEqual(diversity_scores[i], score.item())