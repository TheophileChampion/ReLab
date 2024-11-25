import os
import gymnasium as gym
import ale_py
import torch


def initialize(agent_name, env_name, seed, data_directory=None):
    """
    Initialize the 'benchmarks' package.
    :param agent_name: the agent name
    :param env_name: the environment name
    :param seed: the random seed
    :param data_directory: the path where all the data must be stored
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

    # Set the environment variables: "CHECKPOINT_DIRECTORY", "TENSORBOARD_DIRECTORY" and "DEMO_DIRECTORY".
    suffix = env_name.replace("ALE/", "") + os.sep + agent_name + os.sep + f"{seed}" + os.sep
    os.environ["CHECKPOINT_DIRECTORY"] = os.path.join(os.environ["DATA_DIRECTORY"], "saves", suffix)
    os.environ["TENSORBOARD_DIRECTORY"] = os.path.join(os.environ["DATA_DIRECTORY"], "runs", suffix)
    os.environ["DEMO_DIRECTORY"] = os.path.join(os.environ["DATA_DIRECTORY"], "demos", suffix)

    # Register the Atari environments.
    gym.register_envs(ale_py)

def device():
    """
    Retrieves the device on which the computation should be performed.
    :return: the device
    """
    return torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else "cpu")

def config():
    """
    Retrieves the benchmark configuration.
    :return: the configuration
    """
    return {
        "max_n_steps": 10000000,
        "checkpoint_frequency": 500000,
        "tensorboard_log_interval": 1,
    }
