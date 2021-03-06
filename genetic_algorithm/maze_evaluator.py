import gym
import torch
import torch.nn as nn

from genetic_algorithm.fitness_evaluator import FitnessEvaluator
from genetic_algorithm.genotype import NetworkGenotype
from environment.environment_manager import EnvironmentManager
from utilities.evaluation_logger import EvaluationLogger

class MazeEvaluator(FitnessEvaluator):
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
                total_reward = 0.
                num_iterations = 0
                
                while not done and (max_iterations is None or num_iterations < max_iterations):
                    if visualize : env.render()
                    action = model(state.unsqueeze(0)).argmax().item()
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    num_iterations += 1
                
                fitness += total_reward / float(num_episodes)
                if self.logger is not None: self.logger.log_data(env)
        
        final_pos = env.maze.robot.position
        env.close()
        return fitness, final_pos