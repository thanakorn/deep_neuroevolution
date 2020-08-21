import gym
import torch
import ray
import numpy as np
import matplotlib.pyplot as plt

from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.network_schema import *
from genetic_algorithm.ga import SimpleGA
from genetic_algorithm.fitness_evaluator import GymFitnessEvaluator
from environment.framestack_env_manager import FrameStackEnvManager

def preprocess(screen):
    screen[screen < 255] = 0
    screen = screen / screen.max()
    screen = np.ascontiguousarray(screen)
    return screen

env_id = 'gym_image_maze:ImageMaze-v0'
num_actions = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
frame_stack_size = 1
num_populations = 100
num_episodes_eval = 1

network_schema = {
                    'conv1': ConvSchema(frame_stack_size, 4, 8, 4),
                    'relu1': ActivationSchema('ReLU'),
                    'conv2': ConvSchema(4, 8, 5, 2),
                    'relu2': ActivationSchema('ReLU'),
                    'flatten': ActivationSchema('Flatten'),
                    'fc1': LinearSchema(288, 32),
                    'relu3': ActivationSchema('ReLU'),
                    'output': LinearSchema(32, num_actions)
                 }

evaluator = GymFitnessEvaluator(FrameStackEnvManager, env_name=env_id, preprocess=preprocess, frame_stack_size=frame_stack_size)
init_populations = [TensorGenotype(network_schema, torch.nn.init.xavier_uniform_) for i in range(num_populations)]
ga = SimpleGA(num_populations=num_populations,fitness_evaluator=evaluator, selection_pressure=0.1, 
              mutation_prob=1.0, mutation_power=0.002, crossover_prob=0.5)

# ray.init()
num_generations = 15
solution, info = ga.run(populations=init_populations, num_generations=num_generations, num_workers=10, max_iterations=125, run_mode='multiprocess', visualize=False)
# ray.shutdown()

controller = solution.to_network()
env = FrameStackEnvManager(env_id, frame_stack_size=1, preprocess=preprocess)

plt.figure()
plt.plot(range(num_generations), info.avg_fitnesses, label='AVG_FITNESS')
plt.plot(range(num_generations), info.max_fitnesses, color='red', label='MAX_FITNESS')
plt.legend()
plt.savefig('image_maze_ga.png')

for i in range(10):
    total_reward = 0
    state = env.reset()
    done = False
    
    while not done:
        env.render()
        action = controller(state.unsqueeze(0)).argmax().item()
        state, reward, done, _ = env.step(action)
        total_reward += reward.item()
        
    print('Total reward : ', total_reward)