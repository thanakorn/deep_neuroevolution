import gym
import torch
import torch.nn as nn
from typing import TypeVar, Generic
from genetic_algorithm.genotype import NetworkGenotype
from environment.environment_manager import EnvironmentManager
from model.genetic_network import GeneticNetwork
from utilities.evaluation_logger import EvaluationLogger
from genetic_algorithm.fitness_evaluator import FitnessEvaluator

class ImageMazeDiversityEvaluator(FitnessEvaluator):
    def __init__(self, env_manger: EnvironmentManager, logger: EvaluationLogger = None, **env_args):
        self.env_manger = env_manger
        self.logger = logger
        self.env_args = env_args
        
    def eval_fitness(self, genotype: NetworkGenotype, max_iterations, num_episodes=1, visualize=False):
        env = self.env_manger(**self.env_args)
        with torch.no_grad():
            model = genotype.to_network().to(env.device)
            fitness = 0.

            for ep in range(num_episodes):
                state = env.reset()
                done = False
                num_iterations = 0
                
                while not done and (max_iterations is None or num_iterations < max_iterations):
                    if visualize : env.render()
                    action = model(state.unsqueeze(0)).argmax().item()
                    state, _, done, _ = env.step(action)
                    num_iterations += 1

                if self.logger is not None: self.logger.log_data(env)
        
        dist_travel = env.maze.dist_from_start()
        env.close()
        return dist_travel