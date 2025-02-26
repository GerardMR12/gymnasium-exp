# Algorithms

These below are the algorithms to be used in this repository:

- Proximal Policy Optimization (PPO)


## Proximal Policy Optimization

This algorithm uses the clipping mechanism to avoid very large updates on the model. It uses the `ClipPPOLoss` class form `torchrl`, as well as some others that perform general computations with very optimized procedures. For instance, the generalized advantage estimate or `GAE`. All of these require a certain communication protocol with `Tensordict` objects.

The data gathering is performed by a self-made class called `MyDataCollectorFromEnv` which takes the information from the `Env` object and handles it to fit the desired properties. This class fully surrounds the environment itself, so it can be updated to fit any kind of environment.