import gym
import torch
import ray
from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.network_schema import *
from genetic_algorithm.ga import SimpleGA
from genetic_algorithm.fitness_evaluator import GymFitnessEvaluator
from environment.environment_manager import EnvironmentManager

ray.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env_id = 'Frostbite-ram-v0'
env = gym.make(env_id)

num_populations = 20
num_episodes_eval = 1
network_schema = {
                  'fc1': LinearSchema(env.observation_space.shape[0], 1024),
                  'relu1': ActivationSchema('ReLU'),
                  'fc2': LinearSchema(1024,256),
                  'relu2': ActivationSchema('ReLU'),
                  'fc3': LinearSchema(256,32),
                  'relu3': ActivationSchema('ReLU'),
                  'output': LinearSchema(32, env.action_space.n)
               }

evaluator = GymFitnessEvaluator(EnvironmentManager, env_name=env_id, device=device)
ga = SimpleGA(num_populations=num_populations,fitness_evaluator=evaluator, 
            selection_pressure=0.1, mutation_prob=0.01, mutation_power=0.02, crossover_prob=0.5)
init_populations = [TensorGenotype(network_schema, torch.nn.init.xavier_normal_) for i in range(num_populations)]

solution = ga.run(populations=init_populations, num_generations=20, max_iterations=2000, num_workers=4, run_mode='multiprocess')