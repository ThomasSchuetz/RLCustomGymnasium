import gymnasium as gym

from custom_env.cartpole_gym import CartpoleGym, env_id
from ray.tune.registry import register_env
register_env(env_id, lambda _: CartpoleGym())

from ray.rllib.algorithms.algorithm import Algorithm
latest_checkpoint_dir = "C:\\Users\\tschu\\ray_results\\PPO_cartpole_2023-04-19_20-44-10hz6alfwf\\checkpoint_000005"
agent = Algorithm.from_checkpoint(latest_checkpoint_dir)
env = gym.make(env_id)
state, info = env.reset()


sum_reward = 0
n_step = 20
for step in range(n_step):
    action = agent.compute_single_action(state)
    state, reward, is_done, is_truncated, info = env.step(action)
    sum_reward += reward
    env.render()
    if is_done:
        print("cumulative reward", sum_reward)
        state, info = env.reset()
        sum_reward = 0