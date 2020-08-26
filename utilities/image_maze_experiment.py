import numpy as np

from genetic_algorithm.network_schema import *

__all__ = ['preprocess', 'get_log_data', 'num_actions', 'frame_stack_size', 'num_episodes_eval', 'network_schema']

num_actions = 5
frame_stack_size = 1
num_episodes_eval = 1

network_schema = {
                    'conv1': ConvSchema(frame_stack_size, 8, 8, 4),
                    'relu1': ActivationSchema('ReLU'),
                    'conv2': ConvSchema(8, 16, 5, 2),
                    'relu2': ActivationSchema('ReLU'),
                    'flatten': ActivationSchema('Flatten'),
                    'fc1': LinearSchema(256, 32),
                    'relu3': ActivationSchema('ReLU'),
                    'output': LinearSchema(32, num_actions)
                 }

def preprocess(screen):
    screen[screen < 255] = 0
    screen = screen / screen.max()
    screen = np.ascontiguousarray(screen)
    return screen

def get_log_data(env):
    return env.maze.robot.position, env.maze.dist_to_goal()