import logging
import os
import random
from os import getcwd
from os.path import abspath, join
from typing import Any, Optional

import ale_py
import gymnasium as gym
import numpy as np
import torch
from relab.cpp.agents.memory import CompressorType as Compressor
from relab.environments.SpritesEnv import SpritesEnv
from relab.helpers.Typing import ConfigInfo, Device


def initialize(
    agent_name: str,
    env_name: str,
    seed: int = None,
    data_directory: str = None,
    paths_only: bool = False,
) -> None:
    """!
    Initialize the 'relab' package.
    @param agent_name: the agent name
    @param env_name: the environment name
    @param seed: the random seed
    @param data_directory: the path where all the data must be stored
    @param paths_only: True to only initialize the framework paths, False otherwise
    """

    # Ensure the data directory is valid.
    if data_directory is not None:
        os.environ["DATA_DIRECTORY"] = data_directory
    elif data_directory is None and "DATA_DIRECTORY" in os.environ.keys():
        data_directory = os.environ["DATA_DIRECTORY"]
    else:
        data_directory = abspath(join(getcwd(), "data")) + os.sep
        os.environ["DATA_DIRECTORY"] = data_directory
        logging.info(f"Using default data directory: {data_directory}")

    # Set the environment variables:
    #  "CHECKPOINT_DIRECTORY", "TENSORBOARD_DIRECTORY", "DEMO_DIRECTORY",
    #  "GRAPH_DIRECTORY", "STATISTICS_DIRECTORY", and "DATASET_DIRECTORY".
    suffix = env_name.replace("ALE/", "") + os.sep
    os.environ["GRAPH_DIRECTORY"] = join(data_directory, "graphs", suffix)
    suffix += agent_name + os.sep
    os.environ["STATISTICS_DIRECTORY"] = join(data_directory, "graphs", suffix)
    if seed is not None:
        suffix += f"{seed}" + os.sep
    os.environ["CHECKPOINT_DIRECTORY"] = join(data_directory, "saves", suffix)
    os.environ["TENSORBOARD_DIRECTORY"] = join(data_directory, "runs", suffix)
    os.environ["DEMO_DIRECTORY"] = join(data_directory, "demos", suffix)
    os.environ["DATASET_DIRECTORY"] = join(data_directory, "datasets")

    # Check whether only the paths should be initialized.
    if paths_only is True:
        return

    # Register the Atari and dSprites environments.
    gym.register_envs(ale_py)
    gym.register(id="Sprites-v5", entry_point=SpritesEnv)

    # Set the random seed of all the framework used.
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def device() -> Device:
    """!
    Retrieves the device on which the computation should be performed.
    @return the device
    """
    cuda = torch.cuda.is_available() and torch.cuda.device_count() >= 1
    return torch.device("cuda" if cuda is True else "cpu")


def config(key: Optional[str] = None, value: Any = None) -> ConfigInfo:
    """!
    Retrieves the benchmark configuration.
    @param key: the key whose value in the configuration must be returned, None if the entire configure is requested
    @param value: an optional value to overwrite the configuration
    @return the configuration or the entry in the configuration corresponding to the key passed as parameters
    """

    # Check for configuration overwrite.
    if value is not None:
        return value

    # The ReLab configuration.
    conf = {
        # Maximum number of training iterations
        "max_n_steps": 50000000,
        # Number of training iterations between two checkpoints
        "checkpoint_frequency": 500000,
        # Number of training iterations between two tensorboard logging
        "tensorboard_log_interval": 5000,
        # Number of frames per observation
        "stack_size": 4,
        # Number of times each action is repeated in the environment
        "frame_skip": 1,
        # Size of the images used by the agent to learn
        "screen_size": 84,
        # True, if in-memory compression must be performed, False otherwise
        "compress_png": True,
        # False, if only the last replay buffer must be saved, True otherwise
        "save_all_replay_buffers": False,
    }

    # Check if the user requested the compression type.
    if key == "compression_type":
        return Compressor.ZLIB if conf["compress_png"] else Compressor.RAW

    # Return the entire configure or the requested value.
    return conf if key is None else conf[key]
