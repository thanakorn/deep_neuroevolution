#%%
import matplotlib.pyplot as plt
from environment.framestack_env_manager import FrameStackEnvManager
from environment.classic_control_env_manager import ClassicControlEnvManager

#%% Test EnvironmentManager
env = FrameStackEnvManager('Breakout-v0')

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
env = ClassicControlEnvManager('CartPole-v0')
env.reset()
raw_screen = env.get_raw_screen()
plt.figure()
plt.title('Raw Screen')
plt.axis('off')
plt.imshow(raw_screen)
plt.show()

processed_screen = env.processed_screen(raw_screen)
plt.figure()
plt.title('Processed Screen')
plt.axis('off')
plt.imshow(processed_screen, cmap=plt.get_cmap('gray'))
plt.show()

state=env.reset()
plt.suptitle('Initial State')
plt.axis('off')
plt.imshow(state, cmap=plt.get_cmap('gray'))
plt.show()

for i in range(25): env.step(env.action_space.sample())
state = env.state()
plt.suptitle('Next State')
plt.axis('off')
plt.imshow(state, cmap=plt.get_cmap('gray'))
plt.show()

env.close()
# %%
