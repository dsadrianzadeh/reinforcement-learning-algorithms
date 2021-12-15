# This module implements the "Random Walk" environment.

from gym import Env
from gym.spaces import Discrete


class RandomWalk(Env):

    def __init__(self):

        self.action_space = Discrete(2)  # 0: go left | 1: go right
        self.observation_space = [i for i in range(7)]  # Terminal <-- A <--> B <--> C <--> D <--> E --> Terminal
        self.initial_state = 3
        self.state = None

    def step(self, action):
        """
        Take action and observe the reward and the next state.

        Actions:
            Num   Action
            0     Go left
            1     Go right

        Rewards:
            +1 when terminating on the extreme right
             0 otherwise
        """

        if action == 0:
            self.state -= 1
            if self.state == min(self.observation_space):
                reward = 0
                done = True
            else:
                reward = 0
                done = False

        elif action == 1:
            self.state += 1
            if self.state == max(self.observation_space):
                reward = 1
                done = True
            else:
                reward = 0
                done = False

        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        """
        Reset the environment for a new episode.
        """

        self.state = self.initial_state
        return self.state
