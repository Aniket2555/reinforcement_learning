import gymnasium as gym
import numpy as np
import pickle as pkl

# Creating the environment
CliffWorld = gym.make("CliffWalking-v0")

# Initializing the Q_table
Q_table = np.zeros(shape=(48, 4))

# Creating policy function
def policy(state, epsilon=0.0):
    action = int(np.argmax(Q_table[state]))

    if np.random.random() <= epsilon:
        action = np.random.randint(0, 4)

    return action

# Parameters
EPISODES = 10000
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9

for episode in range(EPISODES):

    done = False
    total_rewards = 0
    episode_length = 0

    # Starting an episode
    state, _ = CliffWorld.reset()

    # Performing action in an episode
    while not done:
        action = policy(state, EPSILON)
        next_state, reward, terminated, truncated, _ = CliffWorld.step(action)
        next_action = policy(next_state, EPSILON)

        Q_table[state][action] += ALPHA * (reward + (GAMMA*Q_table[next_state][next_action]) - Q_table[state][action])

        state = next_state
        action = next_action
        total_rewards += reward
        episode_length += 1
        done = terminated or truncated

    print("episode:", episode, "episode_len:", episode_length, "total_reward:", total_rewards)

pkl.dump(Q_table, open('Sarsa_Q_Table.pkl', 'wb'))
CliffWorld.close()

