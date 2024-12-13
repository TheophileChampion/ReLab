import collections
import os
from os.path import join

import torch

import benchmarks


def display_checkpoint(agent_name, env_name, seed, checkpoint_index, verbose=False):
    """
    Display the key-value pairs in a checkpoint.
    :param agent_name: the agent name
    :param env_name: the environment name
    :param seed: the random seed
    :param checkpoint_index: the number of training steps corresponding to the checkpoint to load
    :param verbose: True if to display even the weights of the networks, False otherwise
    """

    # Initialize the benchmark.
    benchmarks.initialize(agent_name, env_name, seed)

    # Load the checkpoint.
    checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], f"model_{checkpoint_index}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=benchmarks.device())

    # Display key-value pair in the checkpoint.
    for key, value in checkpoint.items():
        if verbose is False and isinstance(value, collections.OrderedDict):
            value = "[Network Weights]"
        print(f"{key}: {value}")


if __name__ == "__main__":

    # Display the key-value pairs in a checkpoint.
    display_checkpoint(agent_name="DDQN", env_name="ALE/Pong-v5", seed=2, checkpoint_index=0)
