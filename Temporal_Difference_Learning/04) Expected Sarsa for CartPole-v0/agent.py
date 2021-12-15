# This module implements the "Expected Sarsa" algorithm.

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

    def update_values(self, state, action, reward, state_):
        """
        Update the action-value estimates.
        """

        # The probability of taking a random action, be it greedy or non-greedy, in an ε-greedy policy is equal to {ε}.
        # It means that each action, then again be it greedy or non-greedy, is taken with a probability of
        # {ε / num_actions}. The probability of taking a greedy action is equal to {1 - ε}. It means that each greedy
        # action is taken with a probability of {(1 - ε) / num_greedy_actions}. Therefore, each non-greedy action is
        # taken with a probability of {ε / num_actions} and each greedy action is taken with a probability of
        # {((1 - ε) / num_greedy_actions) + (ε / num_actions)}. To recap:
        # π(a non-greedy action | s) = ε / num_actions
        # π(a greedy action | s) = ((1 - ε) / num_greedy_actions) + (ε / num_actions)

        max_q_value = np.max(np.array([self.Q[state_, a] for a in range(self.num_actions)]))
        num_greedy_actions = 0

        for a in range(self.num_actions):
            if self.Q[state_, a] == max_q_value:
                num_greedy_actions += 1

        non_greedy_action_probability = self.epsilon / self.num_actions
        greedy_action_probability = ((1 - self.epsilon) / num_greedy_actions) + (self.epsilon / self.num_actions)
        expected_q = 0

        for a in range(self.num_actions):
            if self.Q[state_, a] == max_q_value:
                expected_q += greedy_action_probability * self.Q[state_, a]
            else:
                expected_q += non_greedy_action_probability * self.Q[state_, a]

        delta = reward + self.gamma * expected_q - self.Q[state, action]  # TD error
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
