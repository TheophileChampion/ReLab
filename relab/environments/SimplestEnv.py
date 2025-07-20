from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Env, spaces
from numpy import ndarray
from relab.helpers.Typing import ActionType, Config, GymStepData


class SimplestALE:
    """!
    A class imitating the ALE to make the simplest environment compatible with the Atari wrappers.
    """

    def __init__(self, env: SimplestEnv) -> None:
        """!
        Constructor.
        @param env: the simplest environment for which the ALE is created
        """

        # @var env
        # Reference to the simplest environment being wrapped.
        self.env = env

    def getScreenGrayscale(self, obs: ndarray) -> None:
        """!
        Copy the gray scale screen corresponding to the current environment state into the observation buffer.
        @param obs: the observation buffer
        """
        frame = self.env.current_frame()
        for x in range(obs.shape[0]):
            for y in range(obs.shape[1]):
                obs[x][y] = frame[x][y][0]

    def getScreenRGB(self, obs: ndarray) -> None:
        """!
        Copy the RGB screen corresponding to the current environment state into the observation buffer.
        @param obs: the observation buffer
        """
        frame = self.env.render()
        for x in range(obs.shape[0]):
            for y in range(obs.shape[1]):
                for z in range(obs.shape[2]):
                    obs[x][y][z] = frame[x][y][z]

    def lives(self) -> int:
        """!
        Retrieve the number of lives the agent currently have.
        @return the number of lives
        """
        return 1


class SimplestEnv(Env):
    """!
    @brief A class implementing the Simplest environment.

    @details
    This environment contains two states:
    - a blue image means that an odd action must be selected (e.g., 1, 3, ,,,)
    - a red image means that an even action must be selected (e.g., 0, 2, ,,,)
    """

    # @var metadata
    # Dictionary specifying the environment's rendering capabilities:
    # - render_modes: List of supported rendering modes (rgb_array only)
    # - render_fps: Frame rate for rendering (30 FPS)
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, **kwargs: Any) -> None:
        """!
        Constructor (compatible with OpenAI gym environment)
        @param kwargs: unused
        """

        # Call the parent constructor.
        super(SimplestEnv, self).__init__()

        # @var np_precision
        # The numpy data type used for observations.
        self.dtype = np.uint8

        # @var action_space
        # The space of possible actions.
        self.action_space = spaces.Discrete(18)

        # @var observation_space
        # The space of possible observations.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=self.dtype
        )

        # @var state
        # The current state of the environment.
        self.state = 0

        # @var actions_fn
        # List of action functions that can be performed in the environment.
        self.actions_fn = [self.left, self.right] * 9

        # @var ale
        # Mock of the Arcade Learning Environment interface for compatibility
        # with Atari wrappers.
        self.ale = SimplestALE(self)

        # Reset the environment.
        self.reset()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Config] = None
    ) -> Tuple[ndarray, Dict]:
        """!
        Reset the state of the environment to an initial state.
        @param seed: the seed used to initialize the pseudo random number generator of the environment's (unused)
        @param options: additional information to specify how the environment is reset (unused)
        @return the first observation
        """
        self.state = np.random.choice(2)
        return self.current_frame(), {}

    def step(self, action: ActionType) -> GymStepData:
        """!
        Execute one time step within the environment.
        @param action: the action to perform
        @return next observation, reward, is the trial done?, is the trial truncated?, information
        """

        # Simulate the action requested by the user.
        if not isinstance(action, int):
            action = action.item()
        if action < 0 or action >= len(self.actions_fn):
            exit(f"Invalid action: {action}.")
        reward = self.actions_fn[action]()
        return self.current_frame(), reward, True, False, {}

    def current_frame(self) -> ndarray:
        """!
        Return the current frame (i.e. the current observation).
        @return the current observation
        """
        image = np.zeros((64, 64, 3))
        if self.state == 0:
            image[:, :, 0] = 255
        else:
            image[:, :, 2] = 255
        return image.astype(np.uint8)

    def render(self) -> ndarray:
        """!
        Render the current frame representing the current state of the environment.
        @return the current frame
        """
        return self.current_frame()

    @staticmethod
    def get_action_meanings() -> List[str]:
        """!
        Retrieve the meaning of the environment's actions.
        @return the meaning of the environment's actions
        """
        return ["NOOP", "FIRE"] + ["Left", "Right"] * 8

    #
    # Actions
    #

    def right(self) -> float:
        """!
        Execute the action "right" in the environment.
        @return false (the object never cross the bottom line when moving left)
        """
        # @cond IGNORED_BY_DOXYGEN
        return -1 if self.state % 2 == 0 else 1
        # @endcond

    def left(self) -> float:
        """!
        Execute the action "left" in the environment.
        @return false (the object never cross the bottom line when moving right)
        """
        # @cond IGNORED_BY_DOXYGEN
        return 1 if self.state % 2 == 0 else -1
        # @endcond
