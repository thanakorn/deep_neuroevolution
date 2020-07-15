# %% 
import gym
import torch

from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.network_schema import *
from genetic_algorithm.ga import SimpleGA
from genetic_algorithm.fitness_evaluator import GymFitnessEvaluator
from environment.environment_manager import EnvironmentManager

num_populations = 100
num_episodes_eval = 1
network_schema = {
                    'fc1': LinearSchema(4, 128),
                    'relu1': ActivationSchema('ReLU'),
                    'fc2': LinearSchema(128,32),
                    'relu2': ActivationSchema('ReLU'),
                    'output': LinearSchema(32, 2)
                 }

evaluator = GymFitnessEvaluator(env_name='CartPole-v0', num_episodes=num_episodes_eval, 
                                max_iterations=500, visualize=True)
ga = SimpleGA(num_populations=num_populations,fitness_evaluator=evaluator, 
              selection_pressure=0.1, mutation_prob=0.01, mutation_power=0.02, crossover_prob=0.5)
init_populations = [TensorGenotype(network_schema) for i in range(num_populations)]
#%%
solution = ga.run(populations=init_populations, num_generations=10, num_workers=2)

# %%
num_episodes = 10
env = EnvironmentManager('CartPole-v0')
controller = solution.to_network()
for ep in range(num_episodes):
    total_reward = 0
    state = torch.tensor(env.reset(), dtype=torch.float)
    done = False
    
    while not done:
        env.render()
        action = controller(state.unsqueeze(0)).argmax().item()
        state, reward, done, _ = env.step(action)
        state = torch.tensor(state, dtype=torch.float)
        total_reward += reward
        
    print('Total reward : ', total_reward)
    
env.close()
# %%
evaluator.eval_fitness(init_populations[0])

# %%
env = [gym.make('CartPole-v0') for i in range(15)]
for e in env: print(id(e))

# %%
