import pytest
import math

import torch

from benchmarks.agents.memory.FastDataBuffer import FastDataBuffer as DataBuffer
from benchmarks.agents.memory.ReplayBuffer import Experience


class TestDataBuffer:
    """
    Class testing the data buffer.
    """

    @staticmethod
    def n_steps_reward(r_0, increment, gamma, n):
        r = 0
        for i in range(n):
            r += math.pow(gamma, i) * (r_0 + i * increment)
        return r

    @pytest.mark.parametrize("capacity, n_steps, gamma", [
        (5, 1, 1), (5, 1, 0.9), (5, 2, 1), (5, 2, 0.99),
        (6, 2, 0.95), (7, 1, 0.5), (8, 3, 0.75), (9, 2, 0.8),
        (5, 6, 0.979), (5, 1, 0.98), (5, 1, 0.999), (9, 8, 0.1),
    ])
    def test_storing_and_retrieval_multiple_episodes(self, capacity, n_steps, gamma):

        # Create a data buffer.
        buffer = DataBuffer(capacity=capacity, n_steps=n_steps, gamma=gamma, initial_priority=1, n_children=10)

        # Create the experiences at time t.
        experiences = [
            Experience(obs=torch.zeros([4, 10, 10]), action=t, reward=t, done=(t == capacity - 1), next_obs=torch.zeros([4, 10, 10]))
            for t in range(capacity)
        ] + [
            Experience(obs=torch.zeros([4, 10, 10]), action=t, reward=t, done=(t == capacity - 1), next_obs=torch.zeros([4, 10, 10]))
            for t in range(capacity - 1)
        ]

        # Create the multistep experiences at time t (experiences expected to be returned by the frame buffer).
        result_experiences = []
        for _ in range(2):
            for t in range(capacity):
                tn = min(t + n_steps, capacity)
                tn1 = min(t + n_steps - 1, capacity - 1)
                result_experiences.append(Experience(
                    obs=torch.zeros([4, 10, 10]), action=t, reward=self.n_steps_reward(t, 1, gamma, tn - t),
                    done=(tn1 == capacity - 1), next_obs=torch.zeros([4, 10, 10])
                ))

        # Fill the buffer with experiences.
        for t in range(capacity):
            buffer.append(experiences[t])

        # Check that experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        actions, rewards, dones = buffer[indices]
        for t in range(capacity):
            assert actions[t] == result_experiences[t].action
            assert abs(rewards[t].item() - result_experiences[t].reward) < 1e-5
            assert dones[t] == result_experiences[t].done

        # Keep pushing experiences to the buffer, effectively replacing all experiences in the frame buffer.
        for t in range(capacity - 1):
            buffer.append(experiences[capacity + t])

        # Check that the new experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        actions, rewards, dones = buffer[indices]
        for t in range(capacity - n_steps + 1):
            assert actions[t] == result_experiences[capacity - n_steps + t].action
            assert abs(rewards[t].item() - result_experiences[capacity - n_steps + t].reward) < 1e-5
            assert dones[t] == result_experiences[capacity - n_steps + t].done

    @pytest.mark.parametrize("capacity, n_steps, gamma", [
        (5, 1, 1), (5, 1, 0.9), (5, 2, 1), (5, 2, 0.99),
        (6, 2, 0.95), (7, 1, 0.5), (8, 3, 0.75), (9, 2, 0.8),
        (5, 6, 0.979), (5, 1, 0.98), (5, 1, 0.999), (9, 8, 0.1),
    ])
    def test_storing_and_retrieval(self, capacity, n_steps, gamma):

        # Create a data buffer.
        buffer = DataBuffer(capacity=capacity, n_steps=n_steps, gamma=gamma, initial_priority=1, n_children=10)

        # Create the experiences at time t.
        experiences = [
            Experience(obs=torch.zeros([4, 10, 10]), action=t, reward=t, done=False, next_obs=torch.zeros([4, 10, 10]))
            for t in range(2 * capacity + n_steps - 1)
        ]

        # Create the multistep experiences at time t (experiences expected to be returned by the data buffer).
        result_experiences = [
            Experience(obs=torch.zeros([4, 10, 10]), action=t, reward=self.n_steps_reward(t, 1, gamma, n_steps), done=False, next_obs=torch.zeros([4, 10, 10]))
            for t in range(2 * capacity)
        ]

        # Fill the buffer with experiences.
        n_experiences = capacity + n_steps - 1
        for t in range(n_experiences):
            buffer.append(experiences[t])

        # Check that experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        actions, rewards, dones = buffer[indices]
        for t in range(capacity):
            assert actions[t] == result_experiences[t].action
            assert abs(rewards[t].item() - result_experiences[t].reward) < 1e-5
            assert dones[t] == result_experiences[t].done

        # Keep pushing experiences to the buffer, effectively replacing all experiences in the frame buffer.
        for t in range(capacity):
            buffer.append(experiences[t + n_experiences])

        # Check that the new experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        actions, rewards, dones = buffer[indices]
        for t in range(capacity):
            assert actions[t] == result_experiences[t + capacity].action
            assert abs(rewards[t].item() - result_experiences[t + capacity].reward) < 1e-5
            assert dones[t] == result_experiences[t + capacity].done
