import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

# Function to create a CartPole environment
def make_env():
    def _init():
        env = gym.make("CartPole-v1")
        return env
    return _init

# Number of parallel environments you want to run
num_envs = 16

# Create a list of environment creation functions
env_fns = [make_env() for _ in range(num_envs)]

# Initialize the vectorized environment
vec_env = AsyncVectorEnv(env_fns)

# Reset all environments simultaneously
observations, infos = vec_env.reset()

# Run a simple loop to interact with the environments
for _ in range(10000):
    # Sample a batch of actions (one for each environment)
    actions = vec_env.action_space.sample()
    
    # Take a step in each environment with the sampled actions
    observations, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    
    # Optionally, handle resets if any environment finished an episode
    if any(terminateds) or any(truncateds):
        observations, infos = vec_env.reset()

# Close the environments when done
vec_env.close()
