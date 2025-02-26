import numpy as np
import cv2
import gymnasium as gym


CliffEnv = gym.make("CliffWalking-v0", render_mode = "ansi")
state, _ = CliffEnv.reset()
done = False

while not done:
    print(CliffEnv.render())
    action = int(np.random.randint(low=0, high=4, size=1))
    print(state, "--->", ["up", "right", "down", "left"][action])
    state, reward, terminated, truncated, _ = CliffEnv.step(action)
    done = terminated or truncated
CliffEnv.close()