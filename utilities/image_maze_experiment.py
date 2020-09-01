import numpy as np
import matplotlib.pyplot as plt

from genetic_algorithm.network_schema import *
from scipy.stats import gaussian_kde

__all__ = ['preprocess', 'get_log_data', 'num_actions', 'frame_stack_size', 'num_episodes_eval', 'network_schema']

num_actions = 5
frame_stack_size = 1
num_episodes_eval = 1

network_schema = {
                    'conv1': ConvSchema(frame_stack_size, 4, 8, 4),
                    'relu1': ActivationSchema('ReLU'),
                    'conv2': ConvSchema(4, 8, 5, 2),
                    'relu2': ActivationSchema('ReLU'),
                    'flatten': ActivationSchema('Flatten'),
                    'fc1': LinearSchema(128, 16),
                    'relu3': ActivationSchema('ReLU'),
                    'output': LinearSchema(16, num_actions)
                 }

def preprocess(screen):
    # screen[screen < 255] = 0
    # screen = screen / screen.max()
    screen = np.ascontiguousarray(screen)
    return screen

def get_log_data(env):
    return env.maze.robot.position, env.maze.dist_to_goal()

def plot_learning_curve(distances_log, num_generations, num_populations):
    avg_dist = [np.mean(distances_log[i:(i + 1) * num_populations]) for i in range(num_generations)]
    max_dist = [np.max(distances_log[i:(i + 1) * num_populations]) for i in range(num_generations)]
    fig, ax = plt.subplots()
    ax.plot(range(num_generations), avg_dist, label='AVG')
    ax.plot(range(num_generations), max_dist, color='red', label='BEST')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Distance to Goal(Negative)')
    ax.legend(loc='lower right')
    ax.set_ylim(top=3)
    ax.hlines(0, 0, num_generations, linestyles='dashed', color='black')
    ax.annotate('Goal', xy=(0,1), xycoords='data')
    return fig, ax

def plot_heatmap(bg, positions, limit):
    heatmap = np.zeros(bg.shape)
    for x, y in positions: heatmap[y][x] = heatmap[y][x] + 1
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(bg)
    ax.imshow(np.clip(heatmap, 0, limit), cmap=plt.cm.Reds, interpolation='catrom', alpha=0.2)
    return fig, ax
    
def plot_final_pos(bg, positions):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(bg)
    ax.grid(False)
    ax.scatter([x for x, _ in positions], [y for _, y in positions], color='red', marker='X', alpha=0.3)
    return fig, ax