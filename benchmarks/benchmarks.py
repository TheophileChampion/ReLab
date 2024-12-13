import os
import random
from os.path import join

import gymnasium as gym
import ale_py
import numpy as np
import torch


def initialize(agent_name, env_name, seed=None, data_directory=None, paths_only=False):
    """
    Initialize the 'benchmarks' package.
    :param agent_name: the agent name
    :param env_name: the environment name
    :param seed: the random seed
    :param data_directory: the path where all the data must be stored
    :param paths_only: True to only initialize the framework paths, False otherwise
    """

    # Check that the data directory has been provided by the user.
    if "DATA_DIRECTORY" not in os.environ.keys() and data_directory is None:
        message = "You must provide the path to the data directory by either " + \
                  "setting the environment variable 'DATA_DIRECTORY', " + \
                  "or passing it as parameter to 'benchmarks.initialize()'."
        raise RuntimeError(message)

    # Set the environment variable "DATA_DIRECTORY" if provided as parameters by the user.
    if data_directory is not None:
        os.environ["DATA_DIRECTORY"] = data_directory

    # Set the environment variables:
    #  "CHECKPOINT_DIRECTORY", "TENSORBOARD_DIRECTORY", "DEMO_DIRECTORY", "GRAPH_DIRECTORY", and "STATISTICS_DIRECTORY".
    suffix = env_name.replace("ALE/", "") + os.sep
    os.environ["GRAPH_DIRECTORY"] = join(os.environ["DATA_DIRECTORY"], "graphs", suffix)
    suffix += agent_name + os.sep
    os.environ["STATISTICS_DIRECTORY"] = join(os.environ["DATA_DIRECTORY"], "graphs", suffix)
    if seed is not None:
        suffix += f"{seed}" + os.sep
    os.environ["CHECKPOINT_DIRECTORY"] = join(os.environ["DATA_DIRECTORY"], "saves", suffix)
    os.environ["TENSORBOARD_DIRECTORY"] = join(os.environ["DATA_DIRECTORY"], "runs", suffix)
    os.environ["DEMO_DIRECTORY"] = join(os.environ["DATA_DIRECTORY"], "demos", suffix)

    # Check whether only the paths should be initialized.
    if paths_only is True:
        return

    # Register the Atari environments.
    gym.register_envs(ale_py)

    # Set the random seed of all the framework used.
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def device():
    """
    Retrieves the device on which the computation should be performed.
    :return: the device
    """
    return torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else "cpu")


def config(key=None):
    """
    Retrieves the benchmark configuration.
    :param key: the key whose value in the configuration must be returned, None if the entire configure is requested
    :return: the configuration or the entry in the configuration corresponding to the key passed as parameters
    """
    conf = {
        "stack_size": 4,  # Number of frames per observation
        "frame_skip": 1,  # Number of times each action is repeated in the environment
        "max_n_steps": 50000000,  # Maximum number of training iterations
        "checkpoint_frequency": 500000,  # Number of training iterations between two checkpoints
        "tensorboard_log_interval": 5000,  # Number of training iterations between two logging of values in tensorboard
    }
    return conf if key is None else conf[key]