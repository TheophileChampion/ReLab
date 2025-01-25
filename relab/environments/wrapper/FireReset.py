from typing import Any, Dict, Tuple

import gymnasium as gym
from gymnasium import Env
from numpy import ndarray


class FireReset(gym.Wrapper):
    """!
    Take action on reset for environments that are fixed until firing.
    @param env: environment to wrap
    """

    def __init__(self, env: Env) -> None:
        """!
        Initialize the wrapper.
        @param env: environment to wrap
        """
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs : Any) -> Tuple[ndarray, Dict]:
        """!
        Reset the environment and take the firing action if it is available in the environment.
        @param kwargs: the keyword arguments to pass to the environment's reset function
        @return the first observation and the corresponding info dictionary
        """
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}
