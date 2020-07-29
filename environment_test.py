#%%
import matplotlib.pyplot as plt
import cv2
import numpy as np
from environment.framestack_env_manager import FrameStackEnvManager
from environment.classic_control_env_manager import ClassicControlEnvManager

img_size = (64, 64)

#%% Test EnvironmentManager
def preprocess(screen):
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    screen = cv2.resize(screen, (64,64), interpolation=cv2.INTER_NEAREST)
    screen = np.ascontiguousarray(screen)
    return screen

env = FrameStackEnvManager('Breakout-v0', preprocess=preprocess)

raw_screen = env.get_raw_screen()
plt.figure()
plt.title('Raw Screen')
plt.axis('off')
plt.imshow(raw_screen)#.transpose(1,2,0))
plt.show()

env.reset()
state = env.state()
plt.suptitle('Initial State')
plt.subplot(141)
plt.axis('off')
plt.imshow(state[0], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.axis('off')
plt.imshow(state[1], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.axis('off')
plt.imshow(state[2], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.axis('off')
plt.imshow(state[3], cmap=plt.get_cmap('gray'))
plt.show()

for i in range(3): env.step(env.action_space.sample())
state = env.state()
plt.suptitle('State')
plt.subplot(141)
plt.axis('off')
plt.imshow(state[0], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.axis('off')
plt.imshow(state[1], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.axis('off')
plt.imshow(state[2], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.axis('off')
plt.imshow(state[3], cmap=plt.get_cmap('gray'))
plt.show()


# %%
def preprocess(screen):
    screen = screen[170:320,:]
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    height, width  = 48, 192
    screen = cv2.resize(screen, (width, height), interpolation=cv2.INTER_NEAREST)
    screen = screen[:,int(width / 2) - int(height / 2):int(width / 2) + int(height / 2)]
    screen[screen < 255] = 0
    screen = screen / screen.max()
    screen = np.ascontiguousarray(screen)
    return screen

env = FrameStackEnvManager('CartPole-v0', preprocess=preprocess)
env.reset()
raw_screen = env.get_raw_screen()
plt.figure()
plt.title('Raw Screen')
plt.axis('off')
plt.imshow(raw_screen)
plt.show()

state=env.reset()
plt.suptitle('Initial State')
plt.subplot(221)
plt.axis('off')
plt.imshow(state[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.axis('off')
plt.imshow(state[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.axis('off')
plt.imshow(state[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.axis('off')
plt.imshow(state[3], cmap=plt.get_cmap('gray'))
plt.show()

for i in range(30): env.step(env.action_space.sample())
state = env.state()
plt.suptitle('Next State')
plt.subplot(221)
plt.axis('off')
plt.imshow(state[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.axis('off')
plt.imshow(state[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.axis('off')
plt.imshow(state[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.axis('off')
plt.imshow(state[3], cmap=plt.get_cmap('gray'))
plt.show()

env.close()
# %%
env = FrameStackEnvManager('CartPole-v0', preprocess=preprocess)
env.reset()
# env.reset()
# raw_screen = env.get_raw_screen()
# plt.imshow(raw_screen)
# plt.figure()
# plt.imshow(preprocess(raw_screen), cmap=plt.get_cmap('gray'))
plt.imshow(env.state()[0], cmap=plt.get_cmap('gray'))
env.close()
# %%

env.state()
# %%
def f(a,b,c):
    print(f'A = {a}, B = {b}, C = {c}')
    


# %%
from multiprocessing import Pool

# %%
params = [(1,2,3), (4,5,6), (7,8,9)]
p = Pool()
p.starmap(f, params)

# %%
