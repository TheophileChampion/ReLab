import torch
from benchmarks.agents.memory.ReplayBuffer import FastReplayBuffer
from benchmarks.agents.memory.ReplayBuffer import Experience


if __name__ == "__main__":

    # Create replay buffer.
    capacity = 1000000
    buffer = FastReplayBuffer(capacity=capacity)

    # Append an experience to the replay buffer.
    for i in range(capacity + 1000000):
        if i == capacity:
            print("buffer is full!")
        obs = torch.zeros([4, 84, 84])
        next_obs = torch.ones([4, 84, 84])
        action = 0
        reward = 1.0
        done = False
        buffer.append(Experience(obs, action, reward, done, next_obs))
    print("End of append!")

    # Sample a batch from the replay buffer.0
    for i in range(capacity + 10):
        batch = buffer.sample()

    # Report loss of previous batch to the replay buffer.
    loss = torch.ones([2])
    loss = buffer.report(loss)
