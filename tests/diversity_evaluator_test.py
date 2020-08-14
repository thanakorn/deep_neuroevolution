import unittest
import torch

from genetic_algorithm.diversity_evaluator import BehaviourDiversityEvaluator
from memory.replay_memory import ReplayMemory
from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.network_schema import *

schema = {
            'conv1': ConvSchema(1,4,3,2),
            'flatten': ActivationSchema('Flatten'),
            'output': LinearSchema(36,3)
         }
memory = ReplayMemory(memory_size=32)
populations = [TensorGenotype(schema) for i in range(7)]

for i in range(32): memory.add(torch.rand(8, 8).unsqueeze(0))
    
class DiversityEvaluatorTest(unittest.TestCase):
    def test_diversity_evaluator_init(self):
        evaluator = BehaviourDiversityEvaluator(memory, 8)
        self.assertEqual(evaluator.replay_memory, memory)
        self.assertEqual(evaluator.num_samples, 8)
        
    def test_diversity_evaluator_compute_trajectory(self):
        evaluator = BehaviourDiversityEvaluator(memory, 8)
        samples = memory.sample(5)
        trajectories = evaluator.compute_trajectory([p.to_network() for p in populations], samples)
        
        for p_i in range(len(populations)):
            p = populations[p_i].to_network()
            q_values = p(samples)
            t = trajectories[p_i]
            for s_i in range(samples.shape[0]):
                self.assertTrue(torch.equal(q_values[s_i].argmax(), t[s_i]))
                
    def test_diversity_evaluator_eval_diversity(self):
        evaluator = BehaviourDiversityEvaluator(memory, 8)
        samples = memory.sample(5)
        diversity_scores = evaluator.eval_diversity(populations)
        self.assertEqual(len(diversity_scores), len(populations))
        
        trajectories = evaluator.compute_trajectory([p.to_network() for p in populations], samples)
        for i in range(trajectories.shape[0]):
            score = 0
            for j in range(i + 1, trajectories.shape[0]):
                score += (trajectories[i] != trajectories[j]).sum()
            self.assertEqual(diversity_scores[i], score.item())