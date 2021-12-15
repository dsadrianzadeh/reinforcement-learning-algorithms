# This module implements the "TD(0)" algorithm.

class Agent:

    def __init__(self, alpha, gamma, state_space, action_space):

        self.alpha = alpha  # step-size parameter
        self.gamma = gamma  # discount-rate parameter (discount factor)
        self.state_space = state_space
        self.action_space = action_space

        self.V = {}
        for s in self.state_space:
            self.V[s] = 0

    def policy(self, state):
        """
        Implement the random action selection policy for the "Random Walk" environment.
        """

        _ = state
        action = self.action_space.sample()
        return action

    def update_values(self, state, reward, state_):
        """
        Update the state-value estimates.
        """

        delta = reward + self.gamma * self.V[state_] - self.V[state]  # TD error
        self.V[state] += self.alpha * delta
