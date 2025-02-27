from gymnasium import Env
import gymnasium as gym

import torch
import numpy as np

from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Assuming RunningStat and NormalizedObservation from previous examples:
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

class DataCollectorFromEnv:
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
                with set_exploration_type(ExplorationType.RANDOM):
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