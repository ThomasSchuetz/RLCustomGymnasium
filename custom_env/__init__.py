from gymnasium.envs.registration import register
from .cartpole_gym import env_id
register(id=env_id, entry_point="custom_env.cartpole_gym:CartpoleGym")