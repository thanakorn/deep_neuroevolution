#%%
import gym
import torch
import ray
import cv2
import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.network_schema import *
from genetic_algorithm.ga import SimpleGA
from genetic_algorithm.fitness_evaluator import GymFitnessEvaluator
from environment.framestack_env_manager import FrameStackEnvManager

def preprocess(screen):
    height, width = 110, 84
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    screen = cv2.resize(screen, (width, height))
    screen = screen[int(height / 2) - int(width / 2):int(height / 2) + int(width / 2),:]
    screen = np.ascontiguousarray(screen)
    return screen

ray.init()

env_id = 'Frostbite-v0'
num_actions = 18
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
frame_stack_size = 4
num_populations = 1000
num_episodes_eval = 5
network_schema = {
                    'conv1': ConvSchema(frame_stack_size, 16, 8, 4),
                    'relu1': ActivationSchema('ReLU'),
                    'conv2': ConvSchema(16, 32, 4, 2),
                    'relu2': ActivationSchema('ReLU'),
                    'flatten': ActivationSchema('Flatten'),
                    'fc1': LinearSchema(2592, 256),
                    'relu3': ActivationSchema('ReLU'),
                    'output': LinearSchema(256, num_actions)
                 }

evaluator = GymFitnessEvaluator(FrameStackEnvManager, env_name=env_id, preprocess=preprocess, frame_stack_size=frame_stack_size)
init_populations = [TensorGenotype(network_schema, torch.nn.init.xavier_normal_) for i in range(num_populations)]
ga = SimpleGA(num_populations=num_populations,fitness_evaluator=evaluator, selection_pressure=0.02, mutation_prob=0.01, mutation_power=0.002, crossover_prob=0.5)

solution = ga.run(populations=init_populations, num_generations=100, num_workers=4, run_mode='multiprocess', max_iterations=5000, visualize=False)
torch.save(solution.to_network().state_dict(), 'atari-cnn.cfg')