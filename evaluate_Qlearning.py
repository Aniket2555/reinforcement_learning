import gymnasium as gym
import numpy as np
import pickle as pkl
import cv2

# Creating the environment
CliffWorld = gym.make("CliffWalking-v0")

def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2

    # Vertical Lines
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), color=(0, 0, 0), thickness=1)

    # Horizontal Lines
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)

    # Cliff Box
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(255, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # Goal
    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    # Start
    # frame = cv2.putText(img, text="S", org=(49 * 0 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
    #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return frame


# puts the agent at a state
def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(indices=state, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img


# Initializing the Q_table
Q_table = pkl.load(open("Q_learning_Q_Table.pkl", "rb"))

# Creating policy function
def policy(state, epsilon=0.0):
    action = int(np.argmax(Q_table[state]))

    if np.random.random() <= epsilon:
        action = np.random.randint(0, 4)

    return action

# Parameters
EPISODES = 10

for episode in range(EPISODES):

    frame = initialize_frame()
    done = False
    # Starting an episode
    state, _ = CliffWorld.reset()

    # Performing action in an episode
    while not done:

        frame2 = put_agent(frame.copy(), state)
        cv2.imshow("Grid world",frame2)
        cv2.waitKey(250)
        action = policy(state)
        next_state, reward, terminated, truncated, _ = CliffWorld.step(action)
        next_action = policy(next_state)

        state = next_state
        action = next_action

        done = terminated or truncated


CliffWorld.close()