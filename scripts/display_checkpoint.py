from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import collections
import os
from os.path import join
import logging

import torch

import relab


def display_checkpoint(agent : str, env : str, seed : int, index : int, verbose : bool = False) -> None:
    """
    Display the key-value pairs in a checkpoint.
    :param agent: the agent name
    :param env: the environment name
    :param seed: the random seed
    :param index: the number of training steps corresponding to the checkpoint to load
    :param verbose: True if to display even the weights of the networks, False otherwise
    """

    # Initialize the benchmark.
    relab.initialize(agent, env, seed)

    # Load the checkpoint.
    checkpoint_path = join(os.environ["CHECKPOINT_DIRECTORY"], f"model_{index}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=relab.device(), weights_only=False)

    # Display key-value pair in the checkpoint.
    for key, value in checkpoint.items():
        if verbose is False and isinstance(value, collections.OrderedDict):
            value = "[Network Weights]"
        logging.info(f"{key}: {value}")


if __name__ == "__main__":

    # Parse the script arguments.
    parser = ArgumentParser(prog="display_checkpoint", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--agent", type=str, default="DuelingDDQN", help="name of the agent whose policy needs to be demonstrated")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="name of the environment on which to demonstrate the agent's policy")
    parser.add_argument("--seed", type=int, default=0, help="random seed to use")
    parser.add_argument("--index", type=int, default=10000000, help="index of the checkpoint to load")
    args = parser.parse_args()

    # Display the key-value pairs in a checkpoint.
    display_checkpoint(agent=args.agent, env=args.env, seed=args.seed, index=args.index)
