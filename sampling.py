import gymnasium as gym
from custom_env.cartpole_gym import env_id

env = gym.make(env_id)
env.reset()
sum_reward = 0
for i in range(env.MAX_STEPS):
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    sum_reward += reward
    if done:
        break

print(sum_reward)
