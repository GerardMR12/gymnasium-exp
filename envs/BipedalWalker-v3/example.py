import gymnasium as gym
import time

# Initialise the environment
env = gym.make("BipedalWalker-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        time.sleep(1)
        observation, info = env.reset()

env.render()
env.close()