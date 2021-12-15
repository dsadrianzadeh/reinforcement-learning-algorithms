import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
from environment import RandomWalk


def calculate_rmse(actual, estimated):
    a = np.array(actual)
    b = np.array(estimated)
    mse = ((a - b)**2).mean()
    return np.sqrt(mse)


env = RandomWalk()
action_space = env.action_space  # Discrete(2) = [0, 1]
state_space = env.observation_space

alpha = 0.05  # step-size parameter
gamma = 1.0  # discount-rate parameter (discount factor)

agent = Agent(alpha, gamma, state_space, action_space)

episodes = 100
true_state_values = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 0]
root_mean_squared_error = []

for episode in range(1, episodes + 1):

    state = env.reset()
    done = False

    steps = 0
    # actions = []

    while not done:

        # env.render()
        action = agent.policy(state)
        state_, reward, done, info = env.step(action)
        agent.update_values(state, reward, state_)
        state = state_

        steps += 1
        # actions.append(action)

    root_mean_squared_error.append(calculate_rmse(true_state_values, list(agent.V.values())))

    if episode % 10 == 0:
        print(f"============ Episode: {episode} ============")
        print(f"Steps: {steps}")
        # print(f"Actions: {actions}")

# env.close()

x_axis = [i for i in range(1, episodes + 1)]
y_axis = root_mean_squared_error

plt.figure(figsize=(12, 6), dpi=100)

plt.title("RMSE per Episode for State-Value Estimates")
plt.xlabel("Episode")
plt.ylabel("RMSE")

plt.plot(x_axis, y_axis)
plt.grid()
plt.show()
