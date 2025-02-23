import numpy as np

class PPOReplayBuffer:
    def __init__(
            self,
            buffer_size,
            observation_shape,
            action_shape,
            gamma: float = 0.99,
            gae_lambda: float = 0.95
        ):
        self._buffer_size = buffer_size
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._ptr = 0
        self._full = False
        
        # Pre-allocate memory for the buffer
        self._observations = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self._actions = np.zeros((buffer_size, *action_shape), dtype=np.float32)  # for discrete actions, consider np.int32
        self._rewards = np.zeros(buffer_size, dtype=np.float32)
        self._dones = np.zeros(buffer_size, dtype=np.bool_)
        self._values = np.zeros(buffer_size, dtype=np.float32)
        self._logprobs = np.zeros(buffer_size, dtype=np.float32)
        
        # For computed advantages and returns
        self._advantages = np.zeros(buffer_size, dtype=np.float32)
        self._returns = np.zeros(buffer_size, dtype=np.float32)

    def add(self, obs, action, reward, done, value, logprob):
        """
        Add a new experience to the buffer.
        
        Parameters:
            obs: observation from the environment.
            action: action taken.
            reward: reward received.
            done: whether the episode terminated.
            value: value estimate for the current state.
            logprob: log probability of the action.
        """
        self._observations[self._ptr] = obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = done
        self._values[self._ptr] = value
        self._logprobs[self._ptr] = logprob
        
        self._ptr += 1
        if self._ptr >= self._buffer_size:
            self._full = True
            self._ptr = 0

    def sample(self):
        """
        Sample a batch of experiences from the buffer.
        
        Returns:
            obs: observations.
            actions: actions.
            rewards: rewards.
            dones: whether the episode terminated.
            values: value estimates.
            logprobs: log probabilities of the actions.
        """
        if self._full:
            idxs = np.random.randint(0, self._buffer_size, size=self._buffer_size)
        else:
            idxs = np.random.randint(0, self._ptr, size=self._ptr)
        
        return (
            self._observations[idxs],
            self._actions[idxs],
            self._rewards[idxs],
            self._dones[idxs],
            self._values[idxs],
            self._logprobs[idxs],
            self._advantages[idxs],
            self._returns[idxs]
        )

    def compute_advantages_and_returns(self, last_value):
        """
        Compute advantages and returns for the entire buffer using GAE.
        
        Parameters:
            last_value: the value estimate of the state following the last step in the rollout.
            last_done: boolean indicating if the final state was terminal.
        """
        last_gae_sum = 0
        for step in reversed(range(self._buffer_size)):
            if step == self._buffer_size - 1:
                next_value = last_value
            else:
                next_value = self._values[step + 1]
            delta = self._rewards[step] + (1 - self._dones[step]) * self._gamma * next_value - self._values[step]
            last_gae_sum = delta + (1 - self._dones[step]) * self._gamma * self._gae_lambda * last_gae_sum
            self._advantages[step] = last_gae_sum

        self._returns = self._advantages + self._values