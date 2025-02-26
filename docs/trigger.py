class MyCollector:
    def __init__(self, total_frames, environment):
        self.total_frames = total_frames
        self.environment = environment  # your custom environment
        self.current_frame = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_frame < self.total_frames:
            # Each call collects a new set of data from the environment.
            data = self.environment.step()  # Assume step() returns tensordict-like data
            self.current_frame += 1
            return data
        else:
            raise StopIteration

# Example usage:
# Let's assume you have an environment that defines a step() method.
class MyEnvironment:
    def __init__(self):
        self.counter = 0

    def step(self):
        # For illustration, each step returns a simple dictionary.
        self.counter += 1
        return {"frame": self.counter, "data": f"data_{self.counter}"}

# Create your environment and collector instances:
env = MyEnvironment()
total_frames = 10  # adjust as needed
collector = MyCollector(total_frames, env)
num_epochs = 3  # number of epochs for processing each batch

# Iterate over the collector and process the data in each epoch.
for i, tensordict_data in enumerate(collector):
    print(f"Iteration {i}: {tensordict_data}")
    for _ in range(num_epochs):
        # Process or learn from tensordict_data
        # For example, a simple print to simulate processing:
        print(f"   Processing data: {tensordict_data}")
