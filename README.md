# RLCustomGymnasium
Example for creating a custom gymnasium for playing with reinforcement learning

# Source of inspiration

This medium post by Paco Nathan:
https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5

Sadly, this post used an old version of Ray, which is no longer compatible. This repo uses gymnasium instead of gym to comply with the current (version 2.3.1) way of modeling a custom environment in Ray.

# Files in this repository

## Custom environment

- `custom_env/__init__.py` &rarr; Register the custom environment, whenever it is imported. Otherwise, each script that imports the environment would have to register it again.
- `custom_env/cartpole_gym.py` &rarr; Implementation of the custom environment. The interface requires these methods:
    - `def reset(self, *, seed=None, options=None)` &rarr; returns: `tuple(state, info)`
    - `def step(self, action)` &rarr; returns: `tuple(state, reward, is_done, is_truncated, info)`
    - `def render(self, mode="human")` &rarr; returns: `void`
    - `def close(self)` &rarr; returns: `void`

## Examples on how to interact with and integrate the custom environment

- `apply_policy.py` &rarr; Run the previously learned policy a few times to illustrate that the agent has learned how to optimize this gym.
- `sampling.py` &rarr; Basic, manual test of how the environment behaves when given arbitrary actions.
- `training.py` &rarr; Train the agent on the custom environment.

# Used open source libraries:

- Ray (incl. TensorFlow)
- Gymnasium
