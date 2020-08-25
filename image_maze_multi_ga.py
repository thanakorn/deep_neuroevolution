import gym
import torch
import ray
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv

from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.network_schema import *
from genetic_algorithm.diversity_promoted_ga import DiversityPromotedGA
from genetic_algorithm.diversity_evaluator import BehaviourDiversityEvaluator
from genetic_algorithm.fitness_evaluator import GymFitnessEvaluator
from environment.framestack_env_manager import FrameStackEnvManager
from utilities.evaluation_logger import EvaluationLogger
from memory.replay_memory import ReplayMemory

sns.set(style='darkgrid')

def preprocess(screen):
    screen[screen < 255] = 0
    screen = screen / screen.max()
    screen = np.ascontiguousarray(screen)
    return screen

def get_log_data(env):
    return env.maze.robot.position

env_id = 'gym_image_maze:ImageMaze-v0'
num_actions = 4
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

ray.init()
eval_logger = EvaluationLogger(get_log_data)
replay_memory = ReplayMemory()
diversity_evaluator = BehaviourDiversityEvaluator(replay_memory, num_samples=16)
evaluator = GymFitnessEvaluator(FrameStackEnvManager, eval_logger, env_name=env_id, preprocess=preprocess, frame_stack_size=frame_stack_size, replay_memory=replay_memory)
init_populations = [TensorGenotype(network_schema, torch.nn.init.xavier_normal_) for i in range(num_populations)]
ga = DiversityPromotedGA(num_populations=num_populations,fitness_evaluator=evaluator, diversity_evaluator=diversity_evaluator, mutation_prob=0.2, mutation_power=0.002, crossover_prob=0.5)
num_generations = 100
solution, info = ga.run(populations=init_populations, num_generations=num_generations, num_workers=4, max_iterations=100, run_mode='multiprocess', visualize=False)

# Learning curve
plt.figure()
plt.plot(range(num_generations), np.clip(info.avg_fitnesses, a_min=-250, a_max=250), label='AVG')
plt.plot(range(num_generations), np.clip(info.max_fitnesses, a_min=-250, a_max=250), color='red', label='MAX')
plt.xlabel('Generation')
plt.ylabel('Evaluation Reward')
plt.legend()
plt.savefig(f'./resources/{env_id.split(":")[1]}_multi_ga.png')

# Heatmap
bg_img = f'{env_id.split(":")[1].split("-")[1]}'
bg = cv.resize(cv.imread(f'./resources/{bg_img}.png', 0), (64, 64))
positions_log = eval_logger.get_data()
heatmap = np.zeros((64, 64))
for x,y in positions_log: heatmap[y][x] = min(heatmap[y][x] + 1, 500)
plt.figure()
plt.axis('off')
plt.imshow(bg)
plt.imshow(heatmap, cmap=plt.cm.Reds, interpolation='gaussian', alpha=0.3)
plt.savefig(f'./resources/{env_id.split(":")[1]}_multi_ga_heatmap.png')

ray.shutdown()

controller = solution.to_network()
env = FrameStackEnvManager(env_id, frame_stack_size=1, preprocess=preprocess)
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