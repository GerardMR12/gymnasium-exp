import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.data import TensorDictReplayBuffer
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from gymnasium import Env
from gymnasium import spaces

from algos.aux.data import MyDataCollectorFromEnv

class PPO():
    """
    Proximal Policy Optimization (PPO) algorithm.
    """
    def __init__(
            self,
            env: Env,
            policy: nn.Module,
            v_function: nn.Module,
            horizon: int = 2048,
            total_steps: int = 10000,
            minibatch_size: int = 64,
            optim_steps: int = 10,
            epsilon: float = 0.2,
            gamma: float = 0.99,
            lmbda: float = 0.95,
            c1: float = 1.0,
            c2: float = 0.01,
            lr: float = 3e-4,
            device: torch.device = torch.device("cpu")
        ):
        ########################### Parameters ###########################
        # General hyperparameters
        self._env = env
        self._policy = policy
        self._v_function = v_function
        self._horizon = horizon
        self._total_steps = total_steps
        self._minibatch_size = minibatch_size
        self._optim_steps = optim_steps

        # Sensitive hyperparamaters
        self._epsilon = epsilon
        self._gamma = gamma
        self._lmbda = lmbda
        self._c1 = c1
        self._c2 = c2

        # Cuda device
        self._device = device
        ########################### Parameters ###########################

        ########################### PPO Core ###########################
        policy_td_module = TensorDictModule(
            module=self._policy,
            in_keys=["observation"],
            out_keys=["loc", "scale"]
        )

        self._actor = ProbabilisticActor(
            module=policy_td_module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": self._env.action_space.low, # e.g. tensor([1., 1., 1.])
                "high": self._env.action_space.high, # e.g. tensor([-1., -1., -1.])
            },
            return_log_prob=True
        )

        self._value_module = ValueOperator(
            module=self._v_function,
            in_keys=["observation"]
        )

        self._advantage_module = GAE(
            gamma=self._gamma,
            lmbda=self._lmbda,
            value_network=self._value_module,
            average_gae=True
        )

        self._collector = MyDataCollectorFromEnv(
            env=self._env,
            policy=self._actor,
            batch_steps=self._horizon,
            total_steps=self._total_steps,
            device=self._device
        )

        self._tensordict_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self._horizon),
            sampler=SamplerWithoutReplacement(),
        )

        self._loss_module = ClipPPOLoss(
            actor_network=self._actor,
            critic_network=self._value_module,
            clip_epsilon=self._epsilon,
            entropy_bonus=True,
            critic_coef=self._c1,
            entropy_coef=self._c2
        )

        self._optimizer = torch.optim.Adam(self._loss_module.parameters(), lr)
        self._max_grad_norm = 1.0 # gradient clipping for the loss module
        self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, self._total_steps // self._horizon, 0.0
        )
        ########################### PPO Core ###########################  

    def learn(self):
        """
        Learn the policy using PPO. Inspired from the torchrl implementation.
        """
        ########################### Tracking ###########################
        self._logs = defaultdict(list)
        self._pbar = tqdm(total=self._total_steps)
        self._eval_str = ""
        ########################### Tracking ###########################
        for i, tensordict_data in enumerate(self._collector):
            # We now have a batch of data in tensordict_data
            for _ in range(self._optim_steps):
                # We'll need an "advantage" signal to make PPO work
                # Compute GAE at each epoch as its value depends on the updated value function
                self._advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self._tensordict_buffer.extend(data_view.cpu())

                ########################### See torchrl implementation ###########################
                for _ in range(self._horizon // self._minibatch_size):
                    minibatch = self._tensordict_buffer.sample(self._minibatch_size)
                    loss_vals = self._loss_module(minibatch.to(self._device))
                    loss_value: torch.Tensor = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self._loss_module.parameters(), self._max_grad_norm)
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                ########################### See torchrl implementation ###########################

            with torch.no_grad(): # set_exploration_type(ExplorationType.DETERMINISTIC), 
                # execute a rollout with the trained policy
                # eval_rollout = env.rollout(1000, policy_module)
                self._logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                self._logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                self._logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {self._logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {self._logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {self._logs['eval step_count'][-1]}"
                )
                del eval_rollout

            # Update the logs
            self._logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            self._pbar.update(tensordict_data.numel())
            self._logs["step_count"].append(tensordict_data["step_count"].max().item())
            self._logs["lr"].append(self._optimizer.param_groups[0]["lr"])

            # Update the learning rate
            self._lr_scheduler.step()

        # Save the logs
        self._pbar.close()

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self._logs["reward"])
        plt.title("training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(self._logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(self._logs["eval reward (sum)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(self._logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.savefig(f"learning_progress.png")
    
    def test(self, n_steps: int = 10000):
        """
        Run the agent in the environment for a specified number of timesteps.
        """
        total_rewards = []
        total_steps = []
        observation, _ = self._env.reset()
        rewards = 0
        step_count = 0
        for _ in range(n_steps):
            action, _, _, _ = self._actor(torch.tensor(observation, dtype=torch.float, device=self._device))
            observation, reward, terminated, truncated, _ = self._env.step(action.detach().cpu().numpy())
            rewards += reward
            step_count += 1
            if terminated or truncated:
                observation, _ = self._env.reset()
                total_rewards.append(rewards)
                total_steps.append(step_count)
                rewards = 0
                step_count = 0

        print(f"Average number of steps per trajectory: {sum(total_steps)/len(total_steps)}")

        return total_rewards
    
    def get_plot(self, rewards: list, trained: bool, window_size_denom: int = 10):
        """
        Plot the smoothed rewards over time.
        """
        smoothed = pd.DataFrame(rewards).rolling(window=int(len(rewards)/window_size_denom)).mean()
        plt.plot(smoothed)
        plt.xlabel("Timesteps")
        plt.ylabel("Rewards")
        plt.title("Rewards over time")
        plt.savefig(f"{"trained" if trained else "untrained"}_policy.png")
        plt.clf()