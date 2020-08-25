import gym
import torch
import torch.nn as nn
from typing import TypeVar, Generic, List
from genetic_algorithm.genotype import NetworkGenotype
from environment.environment_manager import EnvironmentManager
from model.genetic_network import GeneticNetwork
from memory.replay_memory import ReplayMemory
from itertools import count

class DiversityEvaluator:
    def eval_diversity(self, populations: List[NetworkGenotype]) -> List[float]:
        raise NotImplementedError()
    
class BehaviourDiversityEvaluator(DiversityEvaluator):
    def __init__(self, replay_memory: ReplayMemory, num_samples=16):
        self.replay_memory = replay_memory
        self.num_samples = num_samples
        
    def compute_trajectory(self, policies, states):
        q_values = torch.stack([p(states) for p in policies], dim=0)
        trajectories = q_values.argmax(dim=2)
        return trajectories
    
    def eval_diversity(self, populations: List[NetworkGenotype]) -> List[float]:
        state_samples = self.replay_memory.sample(self.num_samples)
        trajectories = self.compute_trajectory([p.to_network() for p in populations], state_samples)
        diversity_scores = [(trajectories != trajectories[i]).sum().item() for i in range(len(populations))]
        return diversity_scores