# For the previous versions of gymnasium:
# pip install gymnasium==0.29.1

# ---------- Imports ----------
import multiprocessing
from collections import defaultdict

import numpy as np
import gymnasium as gym
from gymnasium import Env
import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
# ---------- Imports ----------

class RunningStat:
    def __init__(self, shape, eps=1e-8):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)
        self.eps = eps

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self):
        return self.M2 / self.n if self.n > 0 else np.ones_like(self.mean)

    @property
    def std(self):
        return np.sqrt(self.variance) + self.eps

class NormalizedObservation(gym.ObservationWrapper):
    def __init__(self, env, clip=10.0, update=True):
        super().__init__(env)
        self.clip = clip
        self.update = update
        self.running_stat = RunningStat(self.observation_space.shape)

    def observation(self, obs):
        if self.update:
            self.running_stat.update(obs)
        norm_obs = (obs - self.running_stat.mean) / self.running_stat.std
        norm_obs = np.clip(norm_obs, -self.clip, self.clip)
        return norm_obs

class MyDataCollectorFromEnv:
    def __init__(
            self,
            env: Env,
            policy: any,
            batch_steps: int,
            total_steps: int,
            device: torch.device = torch.device("cpu")
        ):
        # Init variables
        self._env = env
        self._policy = policy
        self._batch_steps = batch_steps
        self._total_steps = total_steps
        self._device = device

        # Additional variables
        self._current_step = 0
        self._traj_id = 0
        self._step_count = 0
        self._curr_traj_r = 0

        # Current observation
        self._current_observation, _ = self._env.reset()

    def __iter__(self):
        """
        Returns the iterator object.
        """
        return self
    
    def __next__(self):
        """
        Collects data from the environment using the policy.
        """
        # Fields from main td
        _action = []
        _done = []
        _loc = []
        _observation = []
        _sample_log_prob = []
        _scale = []
        _step_count = []
        _terminated = []
        _truncated = []
        
        # Fields from collector td
        _collector_traj_id = []

        # Fields from next td
        _next_done = []
        _next_observation = []
        _next_reward = []
        _next_step_count = []
        _next_terminated = []
        _next_truncated = []

        if self._current_step < self._total_steps:
            for _ in range(self._batch_steps):
                loc, scale, action, log_prob = self._policy(torch.tensor(self._current_observation, dtype=torch.float, device=self._device))
                next_observation, reward, terminated, truncated, _ = self._env.step(action.detach().cpu().numpy())

                # Fields from main td
                _action.append(action)
                _done.append(False)
                _loc.append(loc)
                _observation.append(self._current_observation.astype(np.float32))
                _sample_log_prob.append(log_prob)
                _scale.append(scale)
                _step_count.append(self._step_count)
                _terminated.append(False)
                _truncated.append(False)

                # Fields from collector td
                _collector_traj_id.append(self._traj_id)

                # Fields from next td
                _next_done.append(terminated or truncated)
                _next_observation.append(next_observation.astype(np.float32))
                _next_reward.append(np.float32(reward))
                _next_step_count.append(self._step_count + 1)
                _next_terminated.append(terminated)
                _next_truncated.append(truncated)

                if terminated or truncated:
                    self._current_observation, _ = self._env.reset()
                    self._traj_id += 1
                    self._step_count = 0
                else:
                    self._current_observation = next_observation
                    self._step_count += 1

                self._current_step += 1

            return TensorDict({
                "action": torch.stack(_action, dim=0).clone().detach().to(self._device),
                "done": torch.tensor(_done, device=self._device),
                "loc": torch.stack(_loc, dim=0).clone().detach().to(self._device),
                "observation": torch.as_tensor(np.array(_observation), device=self._device),  # if _observation is a list of numpy arrays
                "sample_log_prob": torch.stack(_sample_log_prob, dim=0).clone().detach().to(self._device),
                "scale": torch.stack(_scale, dim=0).clone().detach().to(self._device),
                "step_count": torch.tensor(_step_count, device=self._device),
                "terminated": torch.tensor(_terminated, device=self._device),
                "truncated": torch.tensor(_truncated, device=self._device),
                
                "collector": TensorDict({
                    "traj_ids": torch.tensor(_collector_traj_id, device=self._device)
                }, batch_size=self._batch_steps),

                "next": TensorDict({
                    "done": torch.tensor(_next_done, device=self._device),
                    "observation": torch.as_tensor(np.array(_next_observation), device=self._device),
                    "reward": torch.tensor(_next_reward, device=self._device),
                    "step_count": torch.tensor(_next_step_count, device=self._device),
                    "terminated": torch.tensor(_next_terminated, device=self._device),
                    "truncated": torch.tensor(_next_truncated, device=self._device),
                }, batch_size=self._batch_steps),
            }, batch_size=self._batch_steps)
        else:
            raise StopIteration

# ---------- Hyperparameters ----------
# is_fork = multiprocessing.get_start_method() == "fork"
# device = (
#     torch.device(0)
#     if torch.cuda.is_available() and not is_fork
#     else torch.device("cpu")
# )

device = torch.device("cpu")
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

print_info = False

frames_per_batch = 1000
total_frames = 50000 # for a complete training, bring the number of frames up to 1M

sub_batch_size = 64 # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2 # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4  # entropy loss coefficient (c2 in the paper)
# ---------- Hyperparameters ----------

# ---------- Environment ----------
env_name = "Hopper-v4"
kw_args = {}
base_env = GymEnv(env_name=env_name, device=device, **kw_args)

env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

if print_info:
    print("Normalization constant shape:", env.transform[0].loc.shape)
    print("Observation specs:", env.observation_spec)
    print("Rewards specs:", env.reward_spec)
    print("Input specs:", env.input_spec)
    print("Action specs (as defined by input specs):", env.action_spec)

check_env_specs(env)

if print_info:
    n = 3
    rollout = env.rollout(n)
    print(f"Rollout of {n} steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)
# ---------- Environment ----------

# ---------- Policy ----------
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec_unbatched.space.low,
        "high": env.action_spec_unbatched.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)
# ---------- Policy ----------

# ---------- Value Function ----------
value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)
# ---------- Value Function ----------

if print_info:
    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))

# ---------- Data Collector ----------
collector = SyncDataCollector(
    create_env_fn=env,
    policy=policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device
)
# ---------- Data Collector ----------

# ---------- Replay Buffer ----------
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)
# ---------- Replay Buffer ----------

# ---------- Loss Function ----------
advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)
# ---------- Loss Function ----------

# ---------- Training Loop ----------
logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# Dummy observation to initialize lazy modules in the value network.
dummy_obs = torch.zeros(env.observation_spec["observation"].shape[-1], device=device)
_ = value_module(dummy_obs)

del collector, env

env = gym.make("Hopper-v4")
env = NormalizedObservation(env)

collector = MyDataCollectorFromEnv(
    env=env,
    policy=policy_module,
    batch_steps=frames_per_batch,
    total_steps=total_frames,
    device=device
)

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    # print(f"Iteration {i}: {tensordict_data["observation"]}")
    # from time import sleep
    # sleep(0.1)
    print(f"Observation: {tensordict_data['observation'][0]}")
    print(f"Action: {tensordict_data['action'][0]}")
    print(f"Loc: {tensordict_data['loc'][0]}")
    print(f"Scale: {tensordict_data['scale'][0]}")
    print(f"Sample log prob: {tensordict_data['sample_log_prob'][0]}")
    print(f"Step count: {tensordict_data['step_count'][0]}")
    print(f"Terminated: {tensordict_data['terminated'][0]}")
    print(f"Truncated: {tensordict_data['truncated'][0]}")
    print(f"Collector traj id: {tensordict_data['collector']['traj_ids'][0]}")
    print(f"Next done: {tensordict_data['next']['done'][0]}")
    print(f"Next observation: {tensordict_data['next']['observation'][0]}")
    print(f"Next reward: {tensordict_data['next']['reward'][0]}")
    print(f"Next step count: {tensordict_data['next']['step_count'][0]}")
    print(f"Next terminated: {tensordict_data['next']['terminated'][0]}")
    print(f"Next truncated: {tensordict_data['next']['truncated'][0]}")
    exit()
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        # print(f"   Processing advantage: {tensordict_data}")
        # sleep(5)
        data_view = tensordict_data.reshape(-1)
        # print(f"   Processing data view: {data_view}")
        # sleep(5)
        replay_buffer.extend(data_view.cpu())
        # print(f"   Replay buffer: {replay_buffer}")
        # sleep(5)
        # exit()
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            print([loss_vals["loss_objective"], loss_vals["loss_critic"], loss_vals["loss_entropy"]])
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    # if i % 10 == 0:
    #     # We evaluate the policy once every 10 batches of data.
    #     # Evaluation is rather simple: execute the policy without exploration
    #     # (take the expected value of the action distribution) for a given
    #     # number of steps (1000, which is our ``env`` horizon).
    #     # The ``rollout`` method of the ``env`` can take a policy as argument:
    #     # it will then execute this policy at each step.
    #     with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
    #         # execute a rollout with the trained policy
    #         eval_rollout = env.rollout(1000, policy_module)
    #         # print(f"Eval rollout: {eval_rollout}")
    #         # exit()
    #         logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
    #         logs["eval reward (sum)"].append(
    #             eval_rollout["next", "reward"].sum().item()
    #         )
    #         logs["eval step_count"].append(eval_rollout["step_count"].max().item())
    #         eval_str = (
    #             f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
    #             f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
    #             f"eval step-count: {logs['eval step_count'][-1]}"
    #         )
    #         del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

# ---------- Training Loop ----------

# ---------- Results ----------
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.savefig(f"{env_name}_torchrlPPO.png")
# ---------- Results ----------