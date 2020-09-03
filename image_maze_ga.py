import torch
import ray
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2 as cv

from genetic_algorithm.genotype import TensorGenotype, LayerGenotype
from genetic_algorithm.ga import SimpleGA
from genetic_algorithm.maze_evaluator import MazeEvaluator
from environment.framestack_env_manager import FrameStackEnvManager
from utilities.evaluation_logger import EvaluationLogger
from utilities.image_maze_experiment import preprocess, get_log_data, num_actions, frame_stack_size, num_episodes_eval, network_schema, plot_learning_curve, plot_heatmap, plot_final_pos

sns.set(style='darkgrid')

env_id = 'gym_image_maze:ImageMaze-v3'
num_populations = 20
num_generations = 201

ray.init()
eval_logger = EvaluationLogger(get_log_data)
evaluator = MazeEvaluator(FrameStackEnvManager, eval_logger, env_name=env_id, preprocess=preprocess, frame_stack_size=frame_stack_size)
init_populations = [LayerGenotype(network_schema, torch.nn.init.xavier_normal_) for i in range(num_populations)]
ga = SimpleGA(num_populations=num_populations,fitness_evaluator=evaluator, selection_pressure=0.1, mutation_prob=1.0, mutation_power=0.005, crossover_prob=0.5)
solution, info = ga.run(populations=init_populations, num_generations=num_generations, num_workers=4, max_iterations=100, run_mode='multiprocess', visualize=False)

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
learning_plot.savefig(f'./resources/{env_id.split(":")[1]}_ga.png')

# Heatmap
bg_img = f'{env_id.split(":")[1].split("-")[1]}'
heatmap_plot, ax = plot_heatmap(cv.resize(cv.imread(f'./resources/{bg_img}.png', 0), (48, 48)), positions_log, 200)
heatmap_plot.savefig(f'./resources/{env_id.split(":")[1]}_ga_heatmap.png')

# Last gen position
step = max(1, int((num_generations - 1) / 4))
for i in range(0, num_generations, step):
    bg_img = f'{env_id.split(":")[1].split("-")[1]}'
    last_gen_pos = positions_log[i * num_populations:(i + 1) * num_populations]
    last_gen_heatmap_plot, ax = plot_final_pos(cv.resize(cv.imread(f'./resources/{bg_img}.png', 0), (48, 48)), last_gen_pos)
    last_gen_heatmap_plot.savefig(f'./resources/{env_id.split(":")[1]}_ga_{int(i/step)}_pos.png')
    
controller = solution.to_network()
torch.save(controller.state_dict(), f'./resources/{env_id.split(":")[1]}_ga.params')
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