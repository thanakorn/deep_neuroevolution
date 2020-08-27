import torch
import ray
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2 as cv

from genetic_algorithm.genotype import TensorGenotype
from genetic_algorithm.ga import SimpleGA
from genetic_algorithm.fitness_evaluator import GymFitnessEvaluator
from environment.framestack_env_manager import FrameStackEnvManager
from utilities.evaluation_logger import EvaluationLogger
from utilities.image_maze_experiment import preprocess, get_log_data, num_actions, frame_stack_size, num_episodes_eval, network_schema

sns.set(style='darkgrid')

env_id = 'gym_image_maze:ImageMaze-v2'
num_populations = 100
num_generations = 5

ray.init()
eval_logger = EvaluationLogger(get_log_data)
evaluator = GymFitnessEvaluator(FrameStackEnvManager, eval_logger, env_name=env_id, preprocess=preprocess, frame_stack_size=frame_stack_size)
init_populations = [TensorGenotype(network_schema, torch.nn.init.xavier_normal_) for i in range(num_populations)]
ga = SimpleGA(num_populations=num_populations,fitness_evaluator=evaluator, selection_pressure=0.1, mutation_prob=1.0, mutation_power=0.005, crossover_prob=0.5)
solution, info = ga.run(populations=init_populations, num_generations=num_generations, num_workers=4, max_iterations=25, run_mode='multiprocess', visualize=False)

evaluation_log = eval_logger.get_data()
ray.shutdown()

positions_log = [pos for pos, _ in evaluation_log]
distances_log = [-1 * dis for _, dis in evaluation_log]

# Learning curve
avg_dist = [np.mean(distances_log[i:(i + 1) * num_populations]) for i in range(num_generations)]
max_dist = [np.max(distances_log[i:(i + 1) * num_populations]) for i in range(num_generations)]
plt.figure()
plt.plot(range(num_generations), avg_dist, label='AVG')
plt.plot(range(num_generations), max_dist, color='red', label='BEST')
plt.hlines(0, 0, num_generations, linestyles='dashed', color='black')
plt.hlines(-22, 0, num_generations, linestyles='dashed', color='black')
plt.hlines(-10, 0, num_generations, linestyles='dashed', color='black')
plt.annotate('Trap 1', xy=(0,-21), xycoords='data')
plt.annotate('Trap 2', xy=(0,-9), xycoords='data')
plt.annotate('Goal', xy=(0,1), xycoords='data')
plt.ylim(top=3)
plt.xlabel('Generation')
plt.ylabel('Distance to Goal(Negative)')
plt.legend(loc='lower right')
plt.savefig(f'./resources/{env_id.split(":")[1]}_ga.png')

# Heatmap
bg_img = f'{env_id.split(":")[1].split("-")[1]}'
bg = cv.resize(cv.imread(f'./resources/{bg_img}.png', 0), (48, 48))
heatmap = np.zeros((48, 48))
for x,y in positions_log: heatmap[y][x] = min(heatmap[y][x] + 1, 350)
plt.figure()
plt.axis('off')
plt.imshow(bg)
plt.imshow(heatmap, cmap=plt.cm.Reds, interpolation='gaussian', alpha=0.3)
plt.savefig(f'./resources/{env_id.split(":")[1]}_ga_heatmap.png')

controller = solution.to_network()
torch.save(controller.state_dict(), f'./resources/{env_id.split(":")[1]}_ga.params')
env = FrameStackEnvManager(env_id, frame_stack_size=1, preprocess=preprocess)
for i in range(1):
    total_reward = 0
    state = env.reset()
    done = False
    
    while not done:
        env.render()
        action = controller(state.unsqueeze(0)).argmax().item()
        state, reward, done, _ = env.step(action)
        total_reward += reward.item()
        
    print('Total reward : ', total_reward)