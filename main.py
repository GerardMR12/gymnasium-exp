import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor

import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

from algos.ppo import PPO

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the CartPole environment
env = gym.make("Hopper-v4")

# Create the policy
n_hidden = 128
obs_dim = env.observation_space.shape[0]
action_vocab_size = 3
actor_net = nn.Sequential(
    nn.Linear(obs_dim, n_hidden, device=device),
    nn.Tanh(),
    nn.Linear(n_hidden, n_hidden, device=device),
    nn.Tanh(),
    nn.Linear(n_hidden, n_hidden, device=device),
    nn.Tanh(),
    nn.Linear(n_hidden, 2 * action_vocab_size, device=device),
    NormalParamExtractor(),
)

# Create the value function
value_net = nn.Sequential(
    nn.Linear(obs_dim, n_hidden, device=device),
    nn.Tanh(),
    nn.Linear(n_hidden, n_hidden, device=device),
    nn.Tanh(),
    nn.Linear(n_hidden, n_hidden, device=device),
    nn.Tanh(),
    nn.Linear(n_hidden, 1, device=device),
)

# Create the PPO algorithm
ppo = PPO(
    env=env,
    policy=actor_net,
    v_function=value_net,
    total_steps=1000000,
    device=device
)

# Test the untrained policy
print("Testing the untrained policy...")
ppo.get_plot(ppo.test(), trained=False)

# Train the policy for certain timesteps
print("Policy learning...")
ppo.learn()

# Test the trained policy
print("Testing the trained policy...")
ppo.get_plot(ppo.test(), trained=True)