from typing import Any, List

import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import (AtariPreprocessing, FrameStackObservation,
                                NumpyToTorch)

import relab
from relab.environments.wrapper.FireReset import FireReset


def make(env_name: str, **kwargs: Any) -> Env:
    """!
    Create the environment whose name is passed as parameters.
    @param env_name: the name of the environment to instantiate
    @param kwargs: the keyword arguments
    @return the created environment
    """
    config = relab.config()
    env = gym.make(env_name, full_action_space=True, **kwargs)
    env = FireReset(env)
    env = AtariPreprocessing(
        env=env,
        noop_max=0,
        frame_skip=config["frame_skip"],
        screen_size=config["screen_size"],
        grayscale_obs=True,
        scale_obs=True,
    )
    env = FrameStackObservation(env, config["stack_size"])
    env = NumpyToTorch(env)
    return env


def small_benchmark_atari_games() -> List[str]:
    """!
    Retrieve a list of five Atari game names part of a small Atari benchmark.
    @return the list of Atari game names
    """
    return [
        "ALE/Breakout-v5",
        "ALE/Freeway-v5",
        "ALE/MsPacman-v5",
        "ALE/Pong-v5",
        "ALE/SpaceInvaders-v5",
    ]


def benchmark_atari_games() -> List[str]:
    """!
    Retrieve the list of the names of the 57 Atari games part of the Atari benchmark.
    @return the list of Atari game names
    """
    return small_benchmark_atari_games() + [
        "ALE/Alien-v5",
        "ALE/Amidar-v5",
        "ALE/Assault-v5",
        "ALE/Asterix-v5",
        "ALE/Asteroids-v5",
        "ALE/Atlantis-v5",
        "ALE/BankHeist-v5",
        "ALE/BattleZone-v5",
        "ALE/BeamRider-v5",
        "ALE/Berzerk-v5",
        "ALE/Bowling-v5",
        "ALE/Boxing-v5",
        "ALE/Centipede-v5",
        "ALE/ChopperCommand-v5",
        "ALE/CrazyClimber-v5",
        "ALE/Defender-v5",
        "ALE/DemonAttack-v5",
        "ALE/DoubleDunk-v5",
        "ALE/Enduro-v5",
        "ALE/FishingDerby-v5",
        "ALE/Frostbite-v5",
        "ALE/Gopher-v5",
        "ALE/Gravitar-v5",
        "ALE/Hero-v5",
        "ALE/IceHockey-v5",
        "ALE/Jamesbond-v5",
        "ALE/Kangaroo-v5",
        "ALE/Krull-v5",
        "ALE/KungFuMaster-v5",
        "ALE/MontezumaRevenge-v5",
        "ALE/NameThisGame-v5",
        "ALE/Phoenix-v5",
        "ALE/Pitfall-v5",
        "ALE/PrivateEye-v5",
        "ALE/Qbert-v5",
        "ALE/Riverraid-v5",
        "ALE/RoadRunner-v5",
        "ALE/Robotank-v5",
        "ALE/Seaquest-v5",
        "ALE/Skiing-v5",
        "ALE/Solaris-v5",
        "ALE/StarGunner-v5",
        "ALE/Surround-v5",
        "ALE/Tennis-v5",
        "ALE/TimePilot-v5",
        "ALE/Tutankham-v5",
        "ALE/UpNDown-v5",
        "ALE/Venture-v5",
        "ALE/VideoPinball-v5",
        "ALE/WizardOfWor-v5",
        "ALE/YarsRevenge-v5",
        "ALE/Zaxxon-v5",
    ]


def all_atari_games() -> List[str]:
    """!
    Retrieve the list of all Atari game names.
    @return the list of all Atari game names
    """
    return benchmark_atari_games() + [
        "ALE/Adventure-v5",
        "ALE/AirRaid-v5",
        "ALE/Carnival-v5",
        "ALE/ElevatorAction-v5",
        "ALE/JourneyEscape-v5",
        "ALE/Pooyan-v5",
    ]
