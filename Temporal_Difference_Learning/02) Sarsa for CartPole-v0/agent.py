# This module implements the "Sarsa" algorithm.

import numpy as np


class Agent:

    def __init__(self, alpha, epsilon, gamma, state_space, action_space, num_actions):

        self.alpha = alpha  # step-size parameter
        self.epsilon = epsilon  # probability of taking a random action in an ε-greedy policy
        self.gamma = gamma  # discount-rate parameter (discount factor)
        self.state_space = state_space
        self.action_space = action_space
        self.num_actions = num_actions

        self.Q = {}
        for s in self.state_space:
            for a in range(self.num_actions):
                self.Q[s, a] = 0

    def policy(self, state):
        """
        Implement the ε-greedy action selection policy.
        """

        q_values = np.array([self.Q[state, a] for a in range(self.num_actions)])
        r = np.random.random()

        if r < self.epsilon:
            action = self.action_space.sample()  # random action - exploration
        else:
            action = np.argmax(q_values)  # greedy action - exploitation

        return action

    def update_values(self, state, action, reward, state_, action_):
        """
        Update the action-value estimates.
        """

        delta = reward + self.gamma * self.Q[state_, action_] - self.Q[state, action]  # TD error
        self.Q[state, action] += self.alpha * delta

    def decrement_epsilon(self, episodes):
        """
        Decrease the value of ε after each episode.
        **Usage is optional.**
        """

        # This is to decrease the probability of taking a random action and ensure convergence with ε-greedy policies.
        # This is because as the action-value estimates change toward the optimal values, the need for exploration
        # decreases and more exploitation is needed.

        if self.epsilon > 0:
            self.epsilon -= 1 / episodes
        else:
            self.epsilon = 0
