# Initialize ray
import ray
ray.init(ignore_reinit_error=True)

# Register custom environment
from custom_env.cartpole_gym import CartpoleGym
from ray.tune.registry import register_env
select_env = "cartpole"
register_env(select_env, lambda _: CartpoleGym())

# Train agent
import ray.rllib.agents.ppo as ppo
config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
agent = ppo.PPOTrainer(config, env=select_env)

n_iter = 5
for n in range(n_iter):
    result = agent.train()
    chkpt_file = agent.save()

    print(
        f'{n+1:2d} reward ' +
        f'{result["episode_reward_min"]:6.2f}' +
        f'/{result["episode_reward_mean"]:6.2f}/' +
        f'{result["episode_reward_max"]:6.2f} ' +
        f'len {result["episode_len_mean"]:4.2f} ' +
        f'saved {chkpt_file}'
    )
