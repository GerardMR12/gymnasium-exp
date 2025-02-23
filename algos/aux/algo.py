import torch
import torch.nn as nn
import torch.optim as optim

from gymnasium import Env

class Algorithm():
    def __init__(
            self,
            name: str,
            env: Env,
            policy: any,
            loss_module: any = None,
            optimizer: any = optim.Adam,
            lr: float = 3e-4,
            device: torch.device = torch.device("cpu")
        ):
        self._name = name
        self._env = env
        self._policy = policy # should have a forward method
        self._loss_module = loss_module

        if self._loss_module is not None:
            self._optimizer_policy = optimizer(self._policy.parameters(), lr=lr)
        else:
            self._optimizer_policy = optimizer(self._policy.parameters(), lr=lr)

        self._lr = lr
        self._device = device

    def __str__(self):
        return f"{self._name}"

    def __repr__(self):
        return f"This is the {self._name} algorithm."