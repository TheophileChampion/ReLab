from functools import partial

import pytest
import torch

from benchmarks.agents.memory.FastFrameBuffer import FastFrameBuffer as FrameBuffer
from benchmarks.agents.memory.ReplayBuffer import Experience


class TestFrameBuffer:
    """
    Class testing the frame buffer.
    """

    @staticmethod
    def obs(t, frame_skip, stack_size):
        o_0 = torch.arange(0, stack_size).unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, 84, 84)
        return (o_0 + t * frame_skip) / 255

    @pytest.mark.parametrize("capacity, frame_skip, n_steps, stack_size", [
        (5, 1, 1, 2), (5, 1, 1, 4), (5, 1, 2, 4), (5, 2, 2, 4),
        (6, 2, 2, 4), (7, 3, 1, 2), (8, 1, 3, 4), (9, 1, 2, 1),
        (5, 1, 6, 1), (5, 6, 1, 1), (5, 1, 1, 6), (9, 9, 8, 9),
    ])
    def test_storing_and_retrieval_multiple_episodes(self, capacity, frame_skip, n_steps, stack_size):

        # Create a frame buffer.
        buffer = FrameBuffer(capacity=capacity, frame_skip=frame_skip, n_steps=n_steps, stack_size=stack_size)

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
                result_experiences.append(
                    Experience(obs=obs[t], action=t, reward=t, done=False, next_obs=obs[tn])
                )

        # Fill the buffer with experiences.
        for t in range(capacity):
            buffer.append(experiences[t])

        # Check that experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        obs_t, obs_tn = buffer[indices]
        for t in range(capacity):
            assert torch.all(torch.eq(result_experiences[t].obs, obs_t[t])).item()
            assert torch.all(torch.eq(result_experiences[t].next_obs, obs_tn[t])).item()

        # Keep pushing experiences to the buffer, effectively replacing all experiences in the frame buffer.
        for t in range(capacity - 1):
            buffer.append(experiences[capacity + t])

        # Check that the new experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        obs_t, obs_tn = buffer[indices]
        for t in range(capacity - n_steps + 1):
            assert torch.all(torch.eq(result_experiences[capacity - n_steps + t].obs, obs_t[t])).item()
            assert torch.all(torch.eq(result_experiences[capacity - n_steps + t].next_obs, obs_tn[t])).item()

    @pytest.mark.parametrize("capacity, frame_skip, n_steps, stack_size", [
        (5, 1, 1, 2), (5, 1, 1, 4), (5, 1, 2, 4), (5, 2, 2, 4),
        (6, 2, 2, 4), (7, 3, 1, 2), (8, 1, 3, 4), (9, 1, 2, 1),
        (5, 1, 6, 1), (5, 6, 1, 1), (5, 1, 1, 6), (9, 9, 9, 9),
    ])
    def test_storing_and_retrieval(self, capacity, frame_skip, n_steps, stack_size):

        # Create a frame buffer.
        buffer = FrameBuffer(capacity=capacity, frame_skip=frame_skip, n_steps=n_steps, stack_size=stack_size)

        # Create the observations at time t.
        get_obs = partial(self.obs, frame_skip=frame_skip, stack_size=stack_size)
        obs = [get_obs(t) for t in range(2 * capacity + n_steps)]

        # Create the experiences at time t.
        experiences = [
            Experience(obs=obs[t], action=t, reward=t, done=False, next_obs=obs[t + 1])
            for t in range(2 * capacity + n_steps - 1)
        ]

        # Create the multistep experiences at time t (experiences expected to be returned by the frame buffer).
        result_experiences = [
            Experience(obs=obs[t], action=t, reward=t, done=False, next_obs=obs[t + n_steps])
            for t in range(2 * capacity)
        ]

        # Fill the buffer with experiences.
        n_experiences = capacity + n_steps - 1
        for t in range(n_experiences):
            buffer.append(experiences[t])

        # Check that experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        obs_t, obs_tn = buffer[indices]
        for t in range(capacity):
            assert torch.all(torch.eq(result_experiences[t].obs, obs_t[t])).item()
            assert torch.all(torch.eq(result_experiences[t].next_obs, obs_tn[t])).item()

        # Keep pushing experiences to the buffer, effectively replacing all experiences in the frame buffer.
        for t in range(capacity):
            buffer.append(experiences[t + n_experiences])

        # Check that the new experiences in the frame buffer are as expected.
        indices = torch.tensor([t for t in range(capacity)])
        obs_t, obs_tn = buffer[indices]
        for t in range(capacity):
            assert torch.all(torch.eq(result_experiences[capacity + t].obs, obs_t[t])).item()
            assert torch.all(torch.eq(result_experiences[capacity + t].next_obs, obs_tn[t])).item()

    def test_encoding_and_decoding(self):

        # Create a frame buffer.
        buffer = FrameBuffer(capacity=8, frame_skip=1, n_steps=1, stack_size=4)

        for i in range(256):

            # Create the i-th frame to encode and decode.
            frame = i / 255 * torch.ones([84, 84], dtype=torch.float)

            # Encode and decode the i-th frame.
            encoded_frame = buffer.encode(frame)
            decoded_frame = buffer.decode(encoded_frame)

            # Check that the initial and decoded frames are identical.
            assert torch.all(torch.eq(frame, decoded_frame)).item()
