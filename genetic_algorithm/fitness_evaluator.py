import torch
import torch.nn as nn
from typing import TypeVar, Generic
from genetic_algorithm.genotype import NetworkGenotype
from model.genetic_network import GeneticNetwork
from itertools import count

T = TypeVar('T', GeneticNetwork, GeneticNetwork)

class FitnessEvaluator():
    def eval_fitness(self, genotype: NetworkGenotype) -> float:
        raise NotImplementedError()
    
class GymFitnessEvaluator():
    def __init__(self, env, num_episodes=1,device='cpu'):
        self.env = env
        self.num_episodes = num_episodes
        self.device = device
        
    def eval_fitness(self, model_type: T, genotype: NetworkGenotype, visualize = False):
        model = model_type(genotype).to(self.device)
        model.eval()
        fitness = 0.
        
        for ep in range(self.num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                if visualize : self.env.render()
                action = model(state.unsqueeze(0)).argmax()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward.item()
            
            fitness += total_reward / float(self.num_episodes)
            
        self.env.close()
        return fitness