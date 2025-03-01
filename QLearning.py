import gymnasium as gym
import numpy as np
import pickle as pkl

# initializing the environment
CliffWorld = gym.make("CliffWalking-v0")

# Initializing Q_table
Q_table = np.zeros(shape=(48, 4))

# CREATING POLICY
def policy(state, epsilon=0.0):
    action = int(np.argmax(Q_table[state]))

    if np.random.random() <= epsilon:
        action = np.random.randint(0, 4)

    return action

# Parameters
EPISODES = 10000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

# Starting the episodes
for episode in range(EPISODES):

    state, _ = CliffWorld.reset()
    done = False
    total_reward = 0
    episode_length = 0
    # taking action according to policy
    action = policy(state, EPSILON)

    while not done:
        next_state, reward, terminated, truncated, _ = CliffWorld.step(action)
        Q_table[state][action] = Q_table[state][action] + ALPHA*(reward + GAMMA*(np.max(Q_table[next_state])) - Q_table[state][action])

        next_action = policy(next_state)
        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1
        done = terminated or truncated
    print("episode_no.", episode, "episode_length", episode_length,"total_reward", total_reward)

pkl.dump(Q_table, open("Q_learning_Q_table.pkl", 'wb'))
CliffWorld.close()



