import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Initialise the environment
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=8), render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.render()
env.close()