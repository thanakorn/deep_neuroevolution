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
    screen = screen[170:320,:]
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    height, width  = 64, 256
    screen = cv2.resize(screen, (width, height))
    screen = screen[:,int(width / 2) - int(height / 2):int(width / 2) + int(height / 2)]
    screen[screen < 255] = 0
    screen = screen / 255
    screen = np.ascontiguousarray(screen)
    return screen

ray.init()

env_id = 'CartPole-v0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
frame_stack_size = 4
num_populations = 100
num_episodes_eval = 5
network_schema = {
                    'conv1': ConvSchema(frame_stack_size, 8, 5, 3),
                    'relu1': ActivationSchema('ReLU'),
                    'conv2': ConvSchema(8, 16, 3, 2),
                    'relu2': ActivationSchema('ReLU'),
                    'flatten': ActivationSchema('Flatten'),
                    'fc1': LinearSchema(1296, 32),
                    'relu3': ActivationSchema('ReLU'),
                    'output': LinearSchema(32, 2)
                 }

evaluator = GymFitnessEvaluator(FrameStackEnvManager, env_name='CartPole-v0', preprocess=preprocess, frame_stack_size=frame_stack_size)
init_populations = [TensorGenotype(network_schema, torch.nn.init.xavier_normal_) for i in range(num_populations)]
ga = SimpleGA(num_populations=num_populations,fitness_evaluator=evaluator, selection_pressure=0.1, mutation_prob=0.01, mutation_power=0.02, crossover_prob=0.5)

solution = ga.run(populations=init_populations, num_generations=20, num_workers=4, run_mode='multiprocess', num_episodes_eval=3, max_iterations=200)