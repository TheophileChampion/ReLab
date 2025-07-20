from typing import Optional

from gymnasium import Env
from relab.agents.AgentInterface import AgentInterface
from relab.helpers.Typing import (
    ActionType,
    AttributeNames,
    Checkpoint,
    Config,
    ObservationType,
)
import matplotlib.pyplot as plt


class Human(AgentInterface):
    """!
    @brief Implements an agent allowing a Human to play in an environment.

    @details
    Controlling the agent in an environment is done by running a demonstration as follows:

    ```poetry run demo --agent Human --env Sprites-v5```
    """

    def __init__(self, n_actions: int = 18, training: bool = False) -> None:
        """!
        Create a human agent.
        @param n_actions: the number of actions available to the agent
        @param training: True if the agent is being trained, False otherwise
        """
        super().__init__(n_actions=n_actions, training=training)

    def step(self, obs: ObservationType) -> ActionType:
        """!
        Select the next action to perform in the environment.
        @param obs: the observation available to make the decision
        @return the next action to perform
        """
        # @cond IGNORED_BY_DOXYGEN
        while True:
            try:
                print("Action to select (integer between 0 and 17 included): ", end="")
                action = int(input())
                return action
            except Exception:
                print()
        # @endcond

    def train(self, env: Env) -> None:
        """!
        Train the agent in the gym environment passed as parameters
        @param env: the gym environment
        """
        raise RuntimeError("Human agents can't be trained, to play, run a demonstration.")

    def demo(self, env: Env, gif_name: str, max_frames: int = 10000) -> None:
        """!
        Allow a human to play in the gym environment passed as parameters
        @param env: the gym environment
        @param max_frames: the maximum number of frames to be played
        """

        # Reset the environment.
        obs, _ = env.reset()

        # Display the human policy and the environment state.
        for t in range(max_frames):

            # Record the frame associated to the current environment state.
            plt.imshow(env.render().numpy())
            plt.show()

            # Execute an action in the environment.
            action = self.step(obs.to(self.device))
            obs, reward, terminated, truncated, _ = env.step(action)
            print(f"reward: {reward}")
            done = terminated or truncated

            # Restart a new episode if the previous one has ended.
            if done:
                obs, _ = env.reset()

        # Close the environment.
        env.close()

    def load(
        self,
        checkpoint_name: str = "",
        buffer_checkpoint_name: str = "",
        attr_names: Optional[AttributeNames] = None,
    ) -> Checkpoint:
        """!
        Load an agent from the filesystem.
        @param checkpoint_name: the name of the agent checkpoint to load
        @param buffer_checkpoint_name: the name of the replay buffer checkpoint to load ("" for default name)
        @param attr_names: a list of attribute names to load from the checkpoint (load all attributes by default)
        @return the loaded checkpoint object
        """
        return None

    def as_dict(self) -> Config:
        """!
        Convert the agent into a dictionary that can be saved on the filesystem.
        @return the dictionary
        """
        return {}

    def save(
        self,
        checkpoint_name: str,
        buffer_checkpoint_name: str = "",
        agent_conf: Optional[Config] = None,
    ) -> None:
        """!
        Save the agent on the filesystem.
        @param checkpoint_name: the name of the checkpoint in which to save the agent
        @param buffer_checkpoint_name: the name of the checkpoint to save the replay buffer ("" for default name)
        @param agent_conf: a dictionary representing the agent's attributes to be saved (for internal use only)
        """
        super().save(checkpoint_name, buffer_checkpoint_name, self.as_dict())
