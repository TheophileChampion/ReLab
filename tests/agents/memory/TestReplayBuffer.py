import math
from functools import partial

import pytest
import torch

from benchmarks import benchmarks
from benchmarks.agents.memory.ReplayBuffer import Experience
from benchmarks.agents.memory.FastReplayBuffer import FastReplayBuffer as ReplayBuffer


class TestReplayBuffer:
    """
    Class testing the replay buffer with support for prioritization and multistep Q-learning.
    """

    @staticmethod
    def obs(t, frame_skip=1, stack_size=4):
        o_0 = torch.arange(0, stack_size).unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, 10, 10)
        return (o_0 + t * frame_skip) / 255

    @staticmethod
    def n_steps_reward(r_0, increment, gamma, n):
        r = 0
        for i in range(n):
            r += math.pow(gamma, i) * (r_0 + i * increment)
        return r

    @pytest.mark.parametrize("capacity, n_steps, gamma", [
        (5, 1, 1), (5, 1, 0.9), (5, 2, 1), (5, 2, 0.99),
        (6, 2, 0.95), (7, 1, 0.5), (8, 3, 0.75), (9, 2, 0.8),
        (5, 1, 0.98), (5, 1, 0.999), (9, 8, 0.1),
    ])
    def test_storing_and_retrieval_multiple_episodes(self, capacity, n_steps, gamma):

        # Create a replay buffer.
        frame_skip = 1
        stack_size = 4
        buffer = ReplayBuffer(capacity=capacity, frame_skip=frame_skip, stack_size=stack_size, m_args={"n_steps": n_steps, "gamma": gamma})

        # Create the observations at time t.
        get_obs = partial(self.obs, frame_skip=frame_skip, stack_size=stack_size)
        obs = [get_obs(t) for t in range(2 * capacity + n_steps)]

        # Create the experiences at time t.
        experiences = [
            Experience(obs=obs[t], action=t, reward=t, done=(t == capacity - 1), next_obs=obs[t + 1])
            for t in range(capacity)
        ] + [
            Experience(obs=obs[t], action=t, reward=t, done=(t == capacity - 1), next_obs=obs[t + 1])
            for t in range(capacity - 1)
        ]

        # Create the multistep experiences at time t (experiences expected to be returned by the frame buffer).
        result_experiences = []
        for _ in range(2):
            for t in range(capacity):
                tn = min(t + n_steps, capacity)
                tn1 = min(t + n_steps - 1, capacity - 1)
                result_experiences.append(Experience(
                    obs=obs[t], action=t, reward=self.n_steps_reward(t, 1, gamma, tn - t),
                    done=(tn1 == capacity - 1), next_obs=obs[tn]
                ))

        # Fill the buffer with experiences.
        for t in range(capacity):
            buffer.append(experiences[t])

        # Check that experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        obs, action, reward, done, next_obs = buffer.get_experiences(indices)
        for t in range(capacity):
            assert torch.all(torch.eq(obs[t].cpu(), result_experiences[t].obs))
            assert action[t].cpu().item() == result_experiences[t].action
            assert abs(reward[t].cpu().item() - result_experiences[t].reward) < 1e-5
            assert done[t].cpu().item() == result_experiences[t].done
            assert torch.all(torch.eq(next_obs[t].cpu(), result_experiences[t].next_obs))

        # Keep pushing experiences to the buffer, effectively replacing all experiences in the frame buffer.
        for t in range(capacity - 1):
            buffer.append(experiences[capacity + t])

        # Check that the new experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        obs, action, reward, done, next_obs = buffer.get_experiences(indices)
        for t in range(capacity - n_steps + 1):
            assert torch.all(torch.eq(obs[t].cpu(), result_experiences[capacity - n_steps + t].obs))
            assert action[t].cpu().item() == result_experiences[capacity - n_steps + t].action
            assert abs(reward[t].cpu().item() - result_experiences[capacity - n_steps + t].reward) < 1e-5
            assert done[t].cpu().item() == result_experiences[capacity - n_steps + t].done
            assert torch.all(torch.eq(next_obs[t].cpu(), result_experiences[capacity - n_steps + t].next_obs))

    @pytest.mark.parametrize("capacity, n_steps, gamma", [
        (5, 1, 1), (5, 1, 0.9), (5, 2, 1), (5, 2, 0.99),
        (6, 2, 0.95), (7, 1, 0.5), (8, 3, 0.75), (9, 2, 0.8),
        (5, 6, 0.979), (5, 1, 0.98), (5, 1, 0.999), (9, 8, 0.1),
    ])
    def test_storing_and_retrieval(self, capacity, n_steps, gamma):

        # Create a replay buffer.
        frame_skip = 1
        stack_size = 4
        buffer = ReplayBuffer(capacity=capacity, frame_skip=frame_skip, stack_size=stack_size, m_args={"n_steps": n_steps, "gamma": gamma})

        # Create the observations at time t.
        get_obs = partial(self.obs, frame_skip=frame_skip, stack_size=stack_size)
        obs = [get_obs(t) for t in range(2 * capacity + n_steps)]

        # Create the experiences at time t.
        experiences = [
            Experience(obs=obs[t], action=t, reward=t, done=False, next_obs=obs[t + 1])
            for t in range(2 * capacity + n_steps - 1)
        ]

        # Create the multistep experiences at time t (experiences expected to be returned by the data buffer).
        result_experiences = [
            Experience(obs=obs[t], action=t, reward=self.n_steps_reward(t, 1, gamma, n_steps), done=False, next_obs=obs[t + n_steps])
            for t in range(2 * capacity)
        ]

        # Fill the buffer with experiences.
        n_experiences = capacity + n_steps - 1
        for t in range(n_experiences):
            buffer.append(experiences[t])

        # Check that experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        obs, action, reward, done, next_obs = buffer.get_experiences(indices)
        for t in range(capacity):
            assert torch.all(torch.eq(obs[t].cpu(), result_experiences[t].obs))
            assert action[t].cpu().item() == result_experiences[t].action
            assert abs(reward[t].cpu().item() - result_experiences[t].reward) < 1e-5
            assert done[t].cpu().item() == result_experiences[t].done
            assert torch.all(torch.eq(next_obs[t].cpu(), result_experiences[t].next_obs))

        # Keep pushing experiences to the buffer, effectively replacing all experiences in the frame buffer.
        for t in range(capacity):
            buffer.append(experiences[t + n_experiences])

        # Check that the new experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        obs, action, reward, done, next_obs = buffer.get_experiences(indices)
        for t in range(capacity):
            assert torch.all(torch.eq(obs[t].cpu(), result_experiences[t + capacity].obs))
            assert action[t].cpu().item() == result_experiences[t + capacity].action
            assert abs(reward[t].cpu().item() - result_experiences[t + capacity].reward) < 1e-5
            assert done[t].cpu().item() == result_experiences[t + capacity].done
            assert torch.all(torch.eq(next_obs[t].cpu(), result_experiences[t + capacity].next_obs))

    @pytest.mark.parametrize("experiences, n_experiences, n_elements", [
        ([], 0, 0),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10]))], 1, 5),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(2)], 2, 6),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(3)], 3, 7),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(10)], 10, 14),
    ])
    def test_len(self, experiences, n_experiences, n_elements):

        # Arrange.
        buffer = ReplayBuffer(stack_size=4)

        # Act.
        for experience in experiences:
            buffer.append(experience)

        # Assert.
        assert len(buffer) == n_experiences

    @pytest.mark.parametrize("p_args, result", [(None, False), ({"initial_priority": 10}, True)])
    def test_prioritized(self, p_args, result):

        # Arrange.
        buffer = ReplayBuffer(p_args=p_args)

        # Assert.
        assert buffer.is_prioritized() == result

    @pytest.mark.parametrize("experiences", [
        ([]),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10]))]),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(2)]),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(3)]),
        ([Experience(torch.zeros([4, 10, 10]), 0, 0.0, False, torch.zeros([4, 10, 10])) for _ in range(10)]),
    ])
    def test_clear(self, experiences):

        # Arrange.
        buffer = ReplayBuffer(stack_size=4)

        # Act.
        for experience in experiences:
            buffer.append(experience)
        buffer.clear()

        # Assert.
        assert len(buffer) == 0

    def test_report(self):

        # Arrange.
        capacity = 4
        buffer = ReplayBuffer(batch_size=2, capacity=capacity, p_args={"initial_priority": 1, "omega_is": 0.5})
        experiences = [
            Experience(
                obs=self.obs(index), action=index, reward=index, done=False, next_obs=self.obs(index + 1)
            ) for index in range(10)
        ]

        # Acts.
        for experience in experiences[0:capacity]:
            buffer.append(experience)
        buffer.sample()
        loss = 2 * torch.ones([2]).to(benchmarks.device())
        loss = buffer.report(loss)

        # Assert.
        for i in range(capacity):
            if i in buffer.get_last_indices():
                assert abs(buffer.get_priority(i) - 2.0) < 0.0001
            else:
                assert buffer.get_priority(i) == 1.0
        for i in range(2):
            assert abs(loss[i].item() - 2.0) < 0.0001
