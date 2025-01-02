from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from os.path import join

import torch

import benchmarks


def update_checkpoint(agent, env, seed, index, updates):
    """
    Update the key-value pairs provided by the user in a checkpoint.
    :param agent: the agent name
    :param env: the environment name
    :param seed: the random seed
    :param index: the number of training steps corresponding to the checkpoint to load
    :param updates: a dictionary of key-value pairs to update in the checkpoint
    """

    # Initialize the benchmark.
    benchmarks.initialize(agent, env, seed)

    # Load the checkpoint.
    checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], f"model_{index}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=benchmarks.device())

    # Update key-value pair in the checkpoint.
    for key, value in updates.items():
        checkpoint[key] = value

    # Save the checkpoint.
    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":

    # Parse the script arguments.
    parser = ArgumentParser(prog="run_demo", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--agent", type=str, default="DuelingDDQN", help="name of the agent whose policy needs to be demonstrated")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="name of the environment on which to demonstrate the agent's policy")
    parser.add_argument("--seed", type=int, default=0, help="random seed to use")
    parser.add_argument("--index", type=int, default=10000000, help="index of the checkpoint to load")
    args = parser.parse_args()

    # Update the key-value pairs provided by the user in a checkpoint.
    update_checkpoint(agent=args.agent, env=args.env, seed=args.seed, index=args.index, updates={"n_actions": 18})
