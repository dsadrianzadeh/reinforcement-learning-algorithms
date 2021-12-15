# This module implements the "Double Q-learning" algorithm.

import numpy as np


class Agent:

    def __init__(self, alpha, epsilon, gamma, state_space, action_space, num_actions):

        self.alpha = alpha  # step-size parameter
        self.epsilon = epsilon  # probability of taking a random action in an ε-greedy policy
        self.gamma = gamma  # discount-rate parameter (discount factor)
        self.state_space = state_space
        self.action_space = action_space
        self.num_actions = num_actions

        self.Q1, self.Q2 = {}, {}
        for s in self.state_space:
            for a in range(self.num_actions):
                self.Q1[s, a] = 0
                self.Q2[s, a] = 0

    def policy(self, state, cond):
        """
        Implement the ε-greedy action selection policy.
        """

        if cond == "0":
            q_values = np.array([self.Q1[state, a] + self.Q2[state, a] for a in range(self.num_actions)])
        elif cond == "1":
            q_values = np.array([self.Q1[state, a] for a in range(self.num_actions)])
        elif cond == "2":
            q_values = np.array([self.Q2[state, a] for a in range(self.num_actions)])

        r = np.random.random()

        if r < self.epsilon:
            action = self.action_space.sample()  # random action - exploration
        else:
            action = np.argmax(q_values)  # greedy action - exploitation

        return action

    def update_values(self, state, action, reward, state_):
        """
        Update the action-value estimates.
        """

        r = np.random.random()

        if r <= 0.5:
            action_ = self.policy(state_, "1")
            delta = reward + self.gamma * self.Q2[state_, action_] - self.Q1[state, action]  # TD error
            self.Q1[state, action] += self.alpha * delta
        else:
            action_ = self.policy(state_, "2")
            delta = reward + self.gamma * self.Q1[state_, action_] - self.Q2[state, action]  # TD error
            self.Q2[state, action] += self.alpha * delta

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
