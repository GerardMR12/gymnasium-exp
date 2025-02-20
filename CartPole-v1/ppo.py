import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, obs_size: int, action_vocab_size: int, n_hidden: int):
        super(Model, self).__init__()
        layers = []
        layers.append(nn.Linear(obs_size, n_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hidden, action_vocab_size))
        layers.append(nn.ReLU())
        layers.append(nn.Softmax(dim=-1))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        # Ensure x is a tensor of type long (if representing indices)
        if not isinstance(x, torch.Tensor):
            x = np.array(x)
            x = torch.tensor(x, dtype=torch.float)
        # Get the probability distribution over actions
        probs = self.seq(x)

        # Create a categorical distribution from the probabilities
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs

    
class VNetwork(nn.Module):
    def __init__(self, obs_size: int, n_hidden: int):
        super(VNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(obs_size, n_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hidden, 1))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        # Ensure x is a tensor of type long (if representing indices)
        if not isinstance(x, torch.Tensor):
            x = np.array(x)
            x = torch.tensor(x, dtype=torch.float)
        return self.seq(x)

if __name__ == "__main__":
    # Initialise the environment
    reward_scale = 0.001
    env = gym.make("CartPole-v1") # , desc=generate_random_map(size=map_size)

    observation, info = env.reset(seed=42) # reset, observation is the state

    torch.autograd.set_detect_anomaly(True) # detect anomaly
    print_info = False

    # Models parameters
    n_hidden = 16
    learning_rate = 5e-4
    load_model = False

    model = Model(env.observation_space.shape[0], env.action_space.n, n_hidden) # NN policy network
    if load_model and os.path.exists("CartPole-v1/model.pt"):
        print("Loading model...")
        model.load_state_dict(torch.load("CartPole-v1/model.pt")) # load the model
    optimizer_pi = optim.Adam(model.parameters(), lr=learning_rate, maximize=True) # Adam optimizer
    model.train()

    v_net = VNetwork(env.observation_space.shape[0], n_hidden) # NN value network
    optimizer_v = optim.Adam(v_net.parameters(), lr=learning_rate) # Adam optimizer
    v_net.train()

    criterion = nn.MSELoss(reduction="mean") # MSE loss as a loss function

    # PPO parameters
    epsilon = 0.3
    discount = 0.99
    gae_param = 0.95

    # Training parameters
    epochs = 10000
    horizon = 100
    optim_steps = 10
    batch_size = 64

    # Losses
    all_losses_pi = []
    all_losses_v = []
    all_rewards = []

    for e in range(epochs):
        observation, info = env.reset() # reset, observation is the state
        done = [] # initialise done
        log_probs = torch.tensor([]) # initialise old_probs
        observations = [] # initialise observations
        rewards = [] # initialise rewards
        delta_t = [] # initialise delta_t

        for _ in range(horizon):
            action, log_prob, probs = model(observation) # run policy
            next_observation, reward, terminated, truncated, info = env.step(action.item()) # take action
            if print_info:
                print(f"    Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

            log_probs = torch.cat([log_probs, torch.tensor([log_prob])])
            with torch.no_grad():
                done += [float(terminated or truncated)] # store done
                observations = observations + [next_observation] # store observation
                rewards += [reward * reward_scale] # store reward
                delta_t += [reward + (1 - float(terminated or truncated)) * discount * v_net(next_observation) - v_net(observation)] # calculate delta_t

            if terminated or truncated:
                observation, info = env.reset()
            else:
                observation = next_observation

        with torch.no_grad():
            T = len(delta_t)
            advantages = torch.zeros(T)
            g = torch.zeros(T)
            last_advantage = 0.0
            last_return = 0.0

            for t in reversed(range(T)):
                # Compute advantage using GAE
                last_advantage = delta_t[t] + discount * gae_param * last_advantage
                advantages[t] = last_advantage # store A(t)
                
                # Compute discounted return
                last_return = rewards[t] + discount * (1 - done[t]) * last_return
                g[t] = last_return # store G(t)

        for _ in range(optim_steps): # PPsO update
            # Optimise the policy
            sampled_indices = np.random.choice(range(T), size=batch_size, replace=False)
            sampled_observations = [observations[i] for i in sampled_indices]
            sampled_log_probs = log_probs[sampled_indices]
            sampled_advantages = advantages[sampled_indices]

            _, new_log_probs, _ = model(observations)
            ratio = torch.exp(new_log_probs - log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)            

            loss_pi = torch.min(ratio * advantages, clipped_ratio * advantages).mean() # clipped surrogate loss
            optimizer_pi.zero_grad()
            loss_pi.backward()
            optimizer_pi.step()

            loss_v: torch.Tensor = criterion(v_net(observations).view(-1), g) # value function loss
            optimizer_v.zero_grad()
            loss_v.backward()
            optimizer_v.step()

            # Save the losses
            all_losses_pi.append(loss_pi.item())
            all_losses_v.append(loss_v.item())

        all_rewards.append(sum(g))

        print(f"CartPole-v1 epoch: {e}, Policy Loss: {all_losses_pi[-1]:.4f}, Value Loss: {all_losses_v[-1]:.4f}, Total Rewards: {all_rewards[-1]}")

    # Draw the surrogate losses
    smoothed = pd.DataFrame(all_losses_pi).rolling(window=int(len(all_losses_pi)/10)).mean()
    plt.plot(smoothed, label="Clipped Surrogate Loss")
    plt.legend()
    plt.savefig("CartPole-v1/policy_loss.png")
    plt.clf()

    # Draw the value losses
    smoothed = pd.DataFrame(all_losses_v).rolling(window=int(len(all_losses_v)/10)).mean()
    plt.plot(smoothed, label="Value Loss")
    plt.legend()
    plt.savefig("CartPole-v1/values_loss.png")
    plt.clf()

    # Draw the rewards
    smoothed = pd.DataFrame(all_rewards).rolling(window=int(len(all_rewards)/10)).mean()
    plt.plot(smoothed, label="Total Rewards")
    plt.legend()
    plt.savefig("CartPole-v1/rewards.png")
    plt.clf()

    # Save the model
    torch.save(model.state_dict(), "CartPole-v1/model.pt")

    env.close()