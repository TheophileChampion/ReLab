import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, NumpyToTorch, AtariPreprocessing

from benchmarks.environments.wrapper.FireReset import FireReset


def make(env_name, **kwargs):
    """
    Create the environment whose name is required as parameters.
    :param env_name: the name of the environment to instantiate
    :param kwargs: the keyword arguments
    :return: the created environment
    """
    env = gym.make(env_name, full_action_space=True, **kwargs)
    env = FireReset(env)
    env = AtariPreprocessing(env=env, noop_max=0, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=True)
    env = FrameStackObservation(env, 4)
    env = NumpyToTorch(env)
    return env
