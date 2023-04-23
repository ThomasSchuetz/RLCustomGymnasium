import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

env_id = "cartpole"

class CartpoleGym(gym.Env):
    LEFT_MIN = 1
    RIGHT_MAX = 10

    MOVE_LEFT = 0
    MOVE_RIGHT = 1

    MAX_STEPS = 10
    REWARD_AWAY = -2
    REWARD_STEP = -1
    REWARD_GOAL = MAX_STEPS

    metadata = {
        "render.modes": ["human"]
    }

    def __init__(self) -> None:
        self.action_space = Discrete(2)
        self.observation_space = Discrete(self.RIGHT_MAX + 1)

        self.goal = int(0.5 * (self.LEFT_MIN + self.RIGHT_MAX - 1))
        self.init_positions = list(range(self.LEFT_MIN, self.RIGHT_MAX))
        self.init_positions.remove(self.goal)

        self.seed()

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        self.position = int(self.np_random.choice(self.init_positions))
        self.count = 0

        self.state = self.position
        self.reward = 0
        self.is_done = False
        self.info = {}

        return self.state, self.info

    def step(self, action):
        if self.count == self.MAX_STEPS:
            self.is_done = True
        else:
            self.count += 1

            if action == self.MOVE_LEFT:
                if self.position == self.LEFT_MIN:
                    # invalid
                    self.reward = self.REWARD_AWAY
                else:
                    self.position -= 1

                if self.position == self.goal:
                    # on goal now
                    self.reward = self.REWARD_GOAL
                    self.is_done = True
                elif self.position < self.goal:
                    # moving away from goal
                    self.reward = self.REWARD_AWAY
                else:
                    # moving toward goal
                    self.reward = self.REWARD_STEP

            elif action == self.MOVE_RIGHT:
                if self.position == self.RIGHT_MAX:
                    # invalid
                    self.reward = self.REWARD_AWAY
                else:
                    self.position += 1

                if self.position == self.goal:
                    # on goal now
                    self.reward = self.REWARD_GOAL
                    self.is_done = True
                elif self.position > self.goal:
                    # moving away from goal
                    self.reward = self.REWARD_AWAY
                else:
                    # moving toward goal
                    self.reward = self.REWARD_STEP

        self.state = self.position
        self.info["dist"] = self.goal - self.position

        return self.state, self.reward, self.is_done, False, self.info

    def render(self, mode="human"):
        print(
            f"position: {self.state:2d}  reward: {self.reward:2d}  info: {self.info}")

    def close(self):
        pass
