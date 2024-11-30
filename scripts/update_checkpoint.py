import os
from os.path import join

import torch

import benchmarks


def update_checkpoint(agent_name, env_name, seed, checkpoint_index, updates):
    """
    Update the key-value pairs provided by the user in a checkpoint.
    :param agent_name: the agent name
    :param env_name: the environment name
    :param seed: the random seed
    :param checkpoint_index: the number of training steps corresponding to the checkpoint to load
    :param updates: a dictionary of key-value pairs to update in the checkpoint
    """

    # Initialize the benchmark.
    benchmarks.initialize(agent_name, env_name, seed)

    # Load the checkpoint.
    checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], f"model_{checkpoint_index}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=benchmarks.device())

    # Update key-value pair in the checkpoint.
    for key, value in updates.items():
        checkpoint[key] = value

    # Save the checkpoint.
    torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":

    # Demonstrate the policy learnt by a reinforcement learning agent on a gym environment.
    update_checkpoint(
        agent_name="DuelingDDQN", env_name="ALE/Pong-v5", seed=0, checkpoint_index=10000000, updates={"n_actions": 18}
    )
