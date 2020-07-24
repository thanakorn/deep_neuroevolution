# %% 
import gym
import torch
from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.network_schema import *
from genetic_algorithm.ga import SimpleGA
from genetic_algorithm.fitness_evaluator import GymFitnessEvaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env_id = 'Frostbite-ram-v0'
env = gym.make(env_id)

num_populations = 30
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

evaluator = GymFitnessEvaluator(env_name=env_id, num_episodes=num_episodes_eval, 
                                max_iterations=2000, device=device)
ga = SimpleGA(num_populations=num_populations,fitness_evaluator=evaluator, 
              selection_pressure=0.1, mutation_prob=0.01, mutation_power=0.02, crossover_prob=0.5)
init_populations = [TensorGenotype(network_schema, torch.nn.init.xavier_normal_) for i in range(num_populations)]

# %%
solution = ga.run(populations=init_populations, num_generations=50, num_workers=2, run_mode='multithread')

# %%
