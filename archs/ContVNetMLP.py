import torch
import torch.nn as nn
import numpy as np

class VNetwork(nn.Module):
    def __init__(self, obs_size: int, n_hidden: int):
        super(VNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(obs_size, n_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hidden, 1))
        self.seq = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for layer in self.seq:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # Ensure x is a tensor of type long (if representing indices)
        if not isinstance(x, torch.Tensor):
            x = np.array(x)
            x = torch.tensor(x, dtype=torch.float)
        return self.seq(x)