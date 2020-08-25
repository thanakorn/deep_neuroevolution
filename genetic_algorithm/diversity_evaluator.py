import gym
import torch
import torch.nn as nn
from typing import TypeVar, Generic, List
from genetic_algorithm.genotype import NetworkGenotype
from environment.environment_manager import EnvironmentManager
from model.genetic_network import GeneticNetwork
from memory.replay_memory import ReplayMemory
from itertools import count
from utilities.ga_helpers import compute_q_values

class DiversityEvaluator:
    def eval_diversity(self, populations: List[NetworkGenotype]) -> List[float]:
        raise NotImplementedError()
    
class TrajectoryDiversityEvaluator(DiversityEvaluator):
    def __init__(self, replay_memory: ReplayMemory, num_samples=16, num_workers=None):
        self.replay_memory = replay_memory
        self.num_samples = num_samples
        self.num_workers = num_workers
        
    def compute_trajectory(self, policies, states):
        q_values = torch.stack(compute_q_values(policies, states, self.num_workers), dim=0)
        trajectories = q_values.argmax(dim=2)
        return trajectories
    
    def eval_diversity(self, populations: List[NetworkGenotype]):
        state_samples = self.replay_memory.sample(self.num_samples)
        policies = [p.to_network() for p in populations]
        trajectories = self.compute_trajectory(policies, state_samples)
        diversity_scores = [(trajectories != trajectories[i]).sum().item() for i in range(len(populations))]
        return diversity_scores
    
class FinalPosDiversityEvaluator(DiversityEvaluator):
    def __init__(self):
        pass
    
    def eval_diversity(self, populations: List[NetworkGenotype]):
        pass