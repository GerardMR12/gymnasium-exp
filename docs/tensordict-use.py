import torch
from tensordict import TensorDict

obs = torch.tensor([[1, 2, 3], [4, 5, 5]])
act = [6, 7]
rew = [8, 9]
done = [10, 11]

next_obs = torch.tensor([[12, 13, 14], [15, 16, 16]])
next_act = [17, 18]
next_rew = [19, 20]
next_done = [21, 22]

next_td = TensorDict({
    "observation": next_obs,
    "action": next_act,
    "reward": next_rew,
    "done": next_done
}, batch_size=2)

td = TensorDict({
    "observation": obs,
    "action": act,
    "reward": rew,
    "done": done,
    "next": next_td
}, batch_size=2)

print(td["observation"])

############################################################################################################
# These are the fields needed to be output from the collector for the PPO algorithm.
# - action: The action taken in the environment.
# - collector: Contains traj_ids.
#   - traj_ids: The trajectory ids.
# - done: The done signal from the environment.
# - loc: The location of the action (mean).
# - next: Contains the following features.
#   - done: The next done signal from the environment.
#   - observation: The next observation of the environment.
#   - reward: The next reward received from the environment.
#   - step_count: The number of steps taken in the environment.
#   - terminated: The terminated signal from the environment.
#   - truncated: The truncated signal from the environment.
# - observation: The observation of the environment.
# - sample_log_prob: The log probability of the action taken in the environment.
# - scale: The scale of the action (std).
# - step_count: The number of steps taken in the environment.
# - terminated: The terminated signal from the environment.
# - truncated: The truncated signal from the environment.
############################################################################################################

############################################################################################################
# These are the fields after going through the GAE module.
# - action: The action taken in the environment.
# - advantage: The advantage signal for the PPO algorithm. ---- NEW ----
# - collector: Contains traj_ids.
#   - traj_ids: The trajectory ids.
# - done: The done signal from the environment.
# - loc: The location of the action (mean).
# - next: Contains the following features.
#   - done: The next done signal from the environment.
#   - observation: The next observation of the environment.
#   - reward: The next reward received from the environment.
#   - state_value: The value of the state. ---- NEW ----
#   - step_count: The number of steps taken in the environment.
#   - terminated: The terminated signal from the environment.
#   - truncated: The truncated signal from the environment.
# - observation: The observation of the environment.
# - sample_log_prob: The log probability of the action taken in the environment.
# - scale: The scale of the action (std).
# - state_value: The value of the state. ---- NEW ----
# - step_count: The number of steps taken in the environment.
# - terminated: The terminated signal from the environment.
# - truncated: The truncated signal from the environment.
# - value_target: The target value for the value function. ---- NEW ----
############################################################################################################