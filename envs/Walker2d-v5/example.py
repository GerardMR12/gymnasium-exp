import gymnasium as gym

# Initialise the environment
env = gym.make("Walker2d-v5", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.render()
env.close()