import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor

import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

from algos.ppo import PPO
from algos.aux.data import NormalizedObservation

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the CartPole environment
env = gym.make("Hopper-v4")
env = NormalizedObservation(env)
test_env = gym.make("Hopper-v4")
test_env = NormalizedObservation(test_env)

# Create the policy
n_hidden = 256
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
    test_env=test_env,
    policy=actor_net,
    v_function=value_net,
    horizon=1000,
    minibatch_size=64,
    optim_steps=10,
    c1=1.0,
    c2=0.0001,
    lr=3e-4,
    device=device
)

# Train the policy for certain timesteps
print("Policy learning...")
ppo.learn(total_steps=5e6)

# Test the policy
print("Policy testing...")
_, _ = ppo.test(1e4)