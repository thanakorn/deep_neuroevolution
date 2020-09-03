import gym
import torch
import torch.nn as nn
import math

from typing import TypeVar, Generic, List
from genetic_algorithm.genotype import NetworkGenotype
from environment.environment_manager import EnvironmentManager
from model.genetic_network import GeneticNetwork
from memory.replay_memory import ReplayMemory
from itertools import count
from utilities.ga_helpers import compute_q_values, calculate_fitnesses

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
    def eval_diversity(self, final_states):
        diversity_scores = [0.] * len(final_states)
        for i in range(len(final_states)):
            ax, ay = final_states[i]
            for j in range(i+1, len(final_states)):
                bx, by = final_states[j]
                d = math.sqrt((ax - bx)**2 + (ay - by)**2)
                diversity_scores[i] += d
                diversity_scores[j] += d
        return diversity_scores
        