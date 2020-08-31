import torch
import ray
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2 as cv

from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.diversity_promoted_ga import DiversityPromotedGA
from genetic_algorithm.diversity_evaluator import TrajectoryDiversityEvaluator
from genetic_algorithm.fitness_evaluator import GymFitnessEvaluator
from environment.framestack_env_manager import FrameStackEnvManager
from utilities.evaluation_logger import EvaluationLogger
from memory.replay_memory import ReplayMemory
from utilities.image_maze_experiment import preprocess, get_log_data, num_actions, frame_stack_size, num_episodes_eval, network_schema, plot_learning_curve, plot_heatmap, plot_final_pos

sns.set(style='darkgrid')

env_id = 'gym_image_maze:ImageMaze-v0'
num_populations = 200
num_generations = 50

ray.init()
replay_memory = ReplayMemory(200000, memory_ratio=0.01)
eval_logger = EvaluationLogger(get_log_data)
reward_evaluator = GymFitnessEvaluator(FrameStackEnvManager, eval_logger, env_name=env_id, preprocess=preprocess, frame_stack_size=frame_stack_size, replay_memory=replay_memory)
diversity_evaluator = TrajectoryDiversityEvaluator(replay_memory=replay_memory, num_samples=8, num_workers=4)
init_populations = [TensorGenotype(network_schema, torch.nn.init.xavier_normal_) for i in range(num_populations)]
ga = DiversityPromotedGA(num_populations=num_populations,fitness_evaluators=reward_evaluator, diversity_evaluator=diversity_evaluator, mutation_prob=1.0, mutation_power=0.005)
solution, info = ga.run(populations=init_populations, num_generations=num_generations, num_workers=4, max_iterations=75, run_mode='multiprocess', visualize=False)

evaluation_log = eval_logger.get_data()
ray.shutdown()

positions_log = [pos for pos, _ in evaluation_log]
distances_log = [-1 * dis for _, dis in evaluation_log]

# Learning curve
learning_plot, ax = plot_learning_curve(distances_log, num_generations, num_populations)
if env_id.endswith('v1'):
    ax.hlines(-22, 0, num_generations, linestyles='dashed', color='black')
    ax.annotate('Trap', xy=(0,-21), xycoords='data')
if env_id.endswith('v2'):
    ax.hlines(-22, 0, num_generations, linestyles='dashed', color='black')
    ax.annotate('Trap 1', xy=(0,-21), xycoords='data')
    ax.hlines(-10, 0, num_generations, linestyles='dashed', color='black')
    ax.annotate('Trap 2', xy=(0,-9), xycoords='data')
learning_plot.savefig(f'./resources/{env_id.split(":")[1]}_multi_ga.png')

# Heatmap
bg_img = f'{env_id.split(":")[1].split("-")[1]}'
heatmap_plot, ax = plot_heatmap(cv.resize(cv.imread(f'./resources/{bg_img}.png', 0), (48, 48)), positions_log, 200)
heatmap_plot.savefig(f'./resources/{env_id.split(":")[1]}_multi_ga_heatmap.png')

# Last gen position
step = max(1, int((num_generations - 1) / 4))
for i in range(0, num_generations, step):
    bg_img = f'{env_id.split(":")[1].split("-")[1]}'
    last_gen_pos = positions_log[i * num_populations:(i + 1) * num_populations]
    last_gen_heatmap_plot, ax = plot_final_pos(cv.resize(cv.imread(f'./resources/{bg_img}.png', 0), (48, 48)), last_gen_pos)
    last_gen_heatmap_plot.savefig(f'./resources/{env_id.split(":")[1]}_multi_ga_{int(i/step)}_pos.png')

controller = solution.to_network()
torch.save(controller.state_dict(), f'./resources/{env_id.split(":")[1]}_multi_ga.params')
env = FrameStackEnvManager(env_id, frame_stack_size=1, preprocess=preprocess)

for i in range(2):
    total_reward = 0
    state = env.reset()
    done = False
    
    while not done:
        env.render()
        action = controller(state.unsqueeze(0)).argmax().item()
        state, reward, done, _ = env.step(action)
        total_reward += reward.item()
        
    print('Total reward : ', total_reward)