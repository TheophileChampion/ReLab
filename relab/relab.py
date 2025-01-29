import os
import random
from os.path import join, isfile, abspath, dirname
from typing import Optional

import invoke
import logging

import gymnasium as gym
import ale_py
import numpy as np
import torch

from relab.helpers.FileSystem import FileSystem
from relab.environments.SpritesEnv import SpritesEnv
from relab.helpers.Typing import Device, ConfigInfo


def initialize(
    agent_name: str, env_name: str, seed: int = None, data_directory : str = None, paths_only: bool = False
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
    if "DATA_DIRECTORY" not in os.environ.keys() and data_directory is None:
        os.environ["DATA_DIRECTORY"] = abspath(join(dirname(__file__), "..", "data")) + os.sep
        logging.info(f"Using default data directory location: {os.environ["DATA_DIRECTORY"]}")

    # Set the environment variable "DATA_DIRECTORY" if provided as parameters by the user.
    if data_directory is not None:
        os.environ["DATA_DIRECTORY"] = data_directory

    # Set the environment variables:
    #  "CHECKPOINT_DIRECTORY", "TENSORBOARD_DIRECTORY", "DEMO_DIRECTORY", "GRAPH_DIRECTORY",
    #  "STATISTICS_DIRECTORY", and "DATASET_DIRECTORY".
    suffix = env_name.replace("ALE/", "") + os.sep
    os.environ["GRAPH_DIRECTORY"] = join(os.environ["DATA_DIRECTORY"], "graphs", suffix)
    suffix += agent_name + os.sep
    os.environ["STATISTICS_DIRECTORY"] = join(os.environ["DATA_DIRECTORY"], "graphs", suffix)
    if seed is not None:
        suffix += f"{seed}" + os.sep
    os.environ["CHECKPOINT_DIRECTORY"] = join(os.environ["DATA_DIRECTORY"], "saves", suffix)
    os.environ["TENSORBOARD_DIRECTORY"] = join(os.environ["DATA_DIRECTORY"], "runs", suffix)
    os.environ["DEMO_DIRECTORY"] = join(os.environ["DATA_DIRECTORY"], "demos", suffix)
    os.environ["DATASET_DIRECTORY"] = join(os.environ["DATA_DIRECTORY"], "datasets")

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


def build_cpp_library_and_wrapper(cpp_library_name : str = "relab", python_module_name : str = "cpp") -> None:
    """!
    Build the C++ shared library and the python module wrapping the library.
    @param cpp_library_name: the name of the shared library to create
    @param python_module_name: the name of the python module
    """

    # Check if the shared libraries already exist.
    build_directory = os.environ["BUILD_DIRECTORY"]
    shared_library = join(build_directory, f"lib{cpp_library_name}.so")
    module_directory = os.environ["CPP_MODULE_DIRECTORY"]
    files = FileSystem.files_in(module_directory, fr"^{python_module_name}.*")
    if isfile(shared_library) and len(files) != 0:
        return

    # Create the shared libraries.
    invoke.run(
        f"mkdir -p {build_directory} && cd {build_directory} && cmake .. && make && cd .. "
        f"&& mv ./build/librelab_wrapper.so {module_directory}/{python_module_name}`python3.12-config --extension-suffix`"
    )


def device() -> Device:
    """!
    Retrieves the device on which the computation should be performed.
    @return the device
    """
    return torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else "cpu")


def config(key : Optional[str] = None) -> ConfigInfo:
    """!
    Retrieves the benchmark configuration.
    @param key: the key whose value in the configuration must be returned, None if the entire configure is requested
    @return the configuration or the entry in the configuration corresponding to the key passed as parameters
    """
    conf = {
        "max_n_steps": 50000000,  # Maximum number of training iterations
        "checkpoint_frequency": 500000,  # Number of training iterations between two checkpoints
        "tensorboard_log_interval": 5000,  # Number of training iterations between two logging of values in tensorboard
        "stack_size": 4,  # Number of frames per observation
        "frame_skip": 1,  # Number of times each action is repeated in the environment
        "screen_size": 84,  # Size of the images used by the agent to learn
        "compress_images": True,  # True, if in-memory compression must be performed, False otherwise
        "save_all_replay_buffers": False,  # False, if only the last replay buffer must be saved, True otherwise
    }
    return conf if key is None else conf[key]
