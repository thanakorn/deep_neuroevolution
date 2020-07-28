import gym
import torch
import ray
from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.network_schema import *
from genetic_algorithm.distributed_ga import DistributedGeneticAlgorithm
from genetic_algorithm.fitness_evaluator import GymFitnessEvaluator

ray.init(num_gpus=torch.cuda.device_count())

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

remote_evaluator = ray.remote(num_gpus=1)(GymFitnessEvaluator)

ga = DistributedGeneticAlgorithm(num_populations=num_populations,fitness_evaluator=remote_evaluator, selection_pressure=0.1, 
                                 mutation_prob=0.01, mutation_power=0.02, crossover_prob=0.5,
                                 env_name=env_id, num_episodes=num_episodes_eval, max_iterations=2000, device=device, visualize=False)

init_populations = [TensorGenotype(network_schema, torch.nn.init.xavier_normal_) for i in range(num_populations)]
solution = ga.run(populations=init_populations, num_generations=20)