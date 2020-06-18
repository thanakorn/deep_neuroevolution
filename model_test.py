#%%
import random
from model.dqn import DQN
from environment.atari_env_manager import AtariEnvManager

#%%
frame_stack_size = 4
env = AtariEnvManager('Breakout-v0', frame_stack_size=frame_stack_size)
dqn = DQN(frame_stack_size, env.action_space.n)

epsilon = 0.1

for ep in range(10):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        env.render()
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = dqn(state.unsqueeze(0)).argmax()
            
        state, reward, done, _ = env.step(action)
        total_reward += reward.item()
    
    print('Total reward : ', total_reward)

env.close()
# %%
