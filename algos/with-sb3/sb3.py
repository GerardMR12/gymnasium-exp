import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Initialize the PPO agent with an MLP policy
model = PPO("MlpPolicy", env, verbose=1)

# Test the untrained model
total_rewards = []
obs, _ = env.reset()
rewards = 0
for _ in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    rewards += reward
    if terminated or truncated:
        obs, _ = env.reset()
        total_rewards.append(rewards)
        rewards = 0

# Train the model for a specified number of timesteps
model.learn(total_timesteps=100000)

# Save the model if needed
model.save("ppo_CartPole")

# To load the model later:
# model = PPO.load("ppo_cartpole", env=env)

# Test the trained model
obs, _ = env.reset()
rewards = 0
for _ in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    rewards += reward
    if terminated or truncated:
        obs, _ = env.reset()
        total_rewards.append(rewards)
        rewards = 0

plt.plot(total_rewards)
plt.xlabel("Timesteps")
plt.ylabel("Rewards")
plt.title("Rewards over time")
plt.savefig("ppo_CartPole_rewards.png")

env.close()
