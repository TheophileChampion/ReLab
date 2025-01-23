import numpy as np
from gymnasium import spaces, Env

from relab.helpers.SpritesDataset import DataSet
import torch.nn.functional as func
import torch

import relab


class SpritesALE:
    """!
    A class imitating the ALE to make the d-sprites environment compatible with the Atari wrappers.
    """

    def __init__(self, env):
        """!
        Constructor.
        @param env: the d-sprites environment for which the ALE is created
        """
        
        ## @var env
        # Reference to the d-sprites environment being wrapped.
        self.env = env

    def getScreenGrayscale(self, obs):
        """!
        Copy the gray scale screen corresponding to the current environment state into the observation buffer.
        @param obs: the observation buffer
        """
        frame = self.env.current_frame()
        for x in range(obs.shape[0]):
            for y in range(obs.shape[1]):
                obs[x][y] = frame[x][y][0]

    def getScreenRGB(self, obs):
        """!
        Copy the RGB screen corresponding to the current environment state into the observation buffer.
        @param obs: the observation buffer
        """
        frame = self.env.render()
        for x in range(obs.shape[0]):
            for y in range(obs.shape[1]):
                for z in range(obs.shape[2]):
                    obs[x][y][z] = frame[x][y][z]

    def lives(self):
        """!
        Retrieve the number of lives the agent currently have.
        @return the number of lives
        """
        return 1


class SpritesEnv(Env):
    """!
    A class containing the code of the dSprites environment adapted from:
    https://github.com/zfountas/deep-active-inference-mc/blob/master/src/game_environment.py
    """

    ## @var metadata
    # Dictionary specifying the environment's rendering capabilities:
    # - render_modes: List of supported rendering modes (rgb_array only)
    # - render_fps: Frame rate for rendering (30 FPS)
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_length=1000, difficulty="hard", **kwargs):
        """!
        Constructor (compatible with OpenAI gym environment)
        @param max_episode_length: the maximum length of an episode
        @param difficulty: the difficulty of the task, i.e., either easy or hard
        @param kwargs: unused
        """

        # Call the parent constructor.
        super(SpritesEnv, self).__init__()

        ## @var np_precision
        # The numpy data type used for observations.
        self.np_precision = np.uint8

        ## @var action_space
        # The space of possible actions.
        self.action_space = spaces.Discrete(18)

        ## @var observation_space
        # The space of possible observations.
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=self.np_precision)

        ## @var images
        # The images of the d-sprites dataset.
        ## @var s_sizes
        # The number of values per latent variable.
        ## @var s_dim
        # The number of latent variables.
        ## @var s_bases
        # The base values used for state indexing.
        self.images, self.s_sizes, self.s_dim, self.s_bases = DataSet.get()

        ## @var state
        # The current state vector of the environment.
        self.state = np.zeros(self.s_dim, dtype=self.np_precision)

        ## @var last_r
        # The last reward received.
        self.last_r = 0.0

        ## @var frame_id
        # Counter for the current frame in the episode.
        self.frame_id = 0

        ## @var max_episode_length
        # Maximum number of frames in an episode.
        self.max_episode_length = max_episode_length

        ## @var actions_fn
        # List of action functions that can be performed in the environment.
        self.actions_fn = [self.idle, self.idle, self.down, self.up, self.left, self.right] + [self.idle] * 12

        ## @var ale
        # A mock of the Arcade Learning Environment interface for compatibility with Atari wrappers.
        self.ale = SpritesALE(self)

        ## @var difficulty
        # The difficulty level of the environment ('easy' or 'hard').
        self.difficulty = difficulty
        if self.difficulty != "hard" and self.difficulty != "easy":
            raise Exception("Invalid difficulty level, must be either 'easy' or 'hard'.")

        # Reset the environment.
        self.reset()

    @staticmethod
    def state_to_one_hot(state):
        """!
        Transform a state into its one hot representation.
        @param state: the state to transform
        @return the one-hot version of the state
        """
        shape = func.one_hot(state[1], 3)
        scale = func.one_hot(state[2], 6)
        orientation = func.one_hot(state[3], 40)
        pos_x = func.one_hot(state[4], 32)
        pos_y = func.one_hot(state[5], 32)
        return torch.cat([shape, scale, orientation, pos_x, pos_y], dim=0).to(torch.float32)

    def get_state(self, one_hot=True):
        """!
        Getter on the current state of the system.
        @param one_hot: True if the outputs must be a concatenation of one hot encoding,
        False if the outputs must be a vector of scalar values
        @return the current state
        """
        state = torch.from_numpy(self.state).to(torch.int64)
        return self.state_to_one_hot(state) if one_hot else self.state

    def reset(self, seed=None, options=None):
        """!
        Reset the state of the environment to an initial state.
        @param seed: the seed used to initialize the pseudo random number generator of the environment's (unused)
        @param options: additional information to specify how the environment is reset (unused)
        @return the first observation
        """
        self.state = np.zeros(self.s_dim, dtype=self.np_precision)
        self.last_r = 0.0
        self.frame_id = 0
        self.reset_hidden_state()
        return self.current_frame(), {}

    def step(self, action):
        """!
        Execute one time step within the environment.
        @param action: the action to perform
        @return next observation, reward, is the trial done?, information
        """

        # Increase the frame index, that count the number of frames since
        # the beginning of the episode.
        self.frame_id += 1

        # Simulate the action requested by the user.
        if not isinstance(action, int):
            action = action.item()
        if action < 0 or action >= len(self.actions_fn):
            exit(f"Invalid action: {action}.")
        done = self.actions_fn[action]()
        if self.difficulty == "easy":
            self.last_r = self.compute_easy_reward()
        if done:
            return self.current_frame(), self.last_r, True, False, {}

        # Make sure the environment is reset if the maximum number of steps in
        # the episode has been reached.
        if self.frame_id >= self.max_episode_length:
            return self.current_frame(), -1.0, True, False, {}
        else:
            return self.current_frame(), self.last_r, False, False, {}

    def s_to_index(self, s):
        """!
        Compute the index of the image corresponding to the state sent as parameter.
        @param s: the state whose index must be computed
        @return the index
        """
        return np.dot(s, self.s_bases).astype(int)

    def current_frame(self):
        """!
        Return the current frame (i.e. the current observation).
        @return the current observation
        """
        image = self.images[self.s_to_index(self.state)].astype(self.np_precision)
        image = np.repeat(image, 3, 2)
        return (image * 255).astype(np.uint8)

    def render(self):
        """!
        Render the current frame representing the current state of the environment.
        @return the current frame
        """
        return self.current_frame()

    def reset_hidden_state(self):
        """!
        Reset the latent state, i.e, sample the a latent state randomly.
        The latent state contains:
         - a color, i.e. white
         - a shape, i.e. square, ellipse, or heart
         - a scale, i.e. 6 values linearly spaced in [0.5, 1]
         - an orientation, i.e. 40 values in [0, 2 pi]
         - a position in X, i.e. 32 values in [0, 1]
         - a position in Y, i.e. 32 values in [0, 1]
        @return the state sampled
        """
        self.state = np.zeros(self.s_dim, dtype=self.np_precision)
        for s_i, s_size in enumerate(self.s_sizes):
            self.state[s_i] = np.random.randint(s_size)

    @staticmethod
    def get_action_meanings():
        """!
        Retrieve the meaning of the environment's actions.
        @return the meaning of the environment's actions
        """
        return ["NOOP", "FIRE", "Down", "Up", "Left", "Right"]

    #
    # Actions
    #

    @staticmethod
    def idle():
        """!
        Execute the action "idle" in the environment.
        @return false (the object never cross the bottom line when it does not move)
        """
        return False

    def down(self):
        """!
        Execute the action "down" in the environment.
        @return true if the object crossed the bottom line
        """
        # @cond IGNORED_BY_DOXYGEN

        # Increase y coordinate.
        self.y_pos += 1.0

        # If the object did not cross the bottom line, return false.
        if self.y_pos < 32:
            return False

        # If the object did cross the bottom line, compute the reward and return true.
        if self.difficulty == "hard":
            self.last_r = self.compute_hard_reward()
        self.y_pos -= 1.0
        return True
        # @endcond

    def up(self):
        """!
        Execute the action "up" in the environment.
        @return false (the object never cross the bottom line when moving up)
        """
        # @cond IGNORED_BY_DOXYGEN
        if self.y_pos > 0:
            self.y_pos -= 1.0
        return False
        # @endcond

    def right(self):
        """!
        Execute the action "right" in the environment.
        @return false (the object never cross the bottom line when moving left)
        """
        # @cond IGNORED_BY_DOXYGEN
        if self.x_pos < 31:
            self.x_pos += 1.0
        return False
        # @endcond

    def left(self):
        """!
        Execute the action "left" in the environment.
        @return false (the object never cross the bottom line when moving right)
        """
        # @cond IGNORED_BY_DOXYGEN
        if self.x_pos > 0:
            self.x_pos -= 1.0
        return False
        # @endcond

    #
    # Reward computation
    #

    def compute_square_reward(self):
        """!
        Compute the obtained by the agent when a square crosses the bottom wall
        @return the reward.
        """
        # @cond IGNORED_BY_DOXYGEN
        if self.x_pos > 15:
            return float(15.0 - self.x_pos) / 16.0
        else:
            return float(16.0 - self.x_pos) / 16.0
        # @endcond

    def compute_non_square_reward(self):
        """!
        Compute the obtained by the agent when an ellipse or heart crosses the bottom wall.
        @return the reward
        """
        # @cond IGNORED_BY_DOXYGEN
        if self.x_pos > 15:
            return float(self.x_pos - 15.0) / 16.0
        else:
            return float(self.x_pos - 16.0) / 16.0
        # @endcond

    def compute_easy_reward(self):
        """!
        Compute the reward obtained by the agent if the environment difficulty is easy.
        @return the reward
        """
        # @cond IGNORED_BY_DOXYGEN
        tx, ty = (0, 31) if self.state[1] < 0.5 else (31, 31)
        return -1.0 + (62 - abs(tx - self.x_pos) - abs(ty - self.y_pos)) / 31.0
        # @endcond

    def compute_hard_reward(self):
        """!
        Compute the reward obtained by the agent if the environment difficulty is hard.
        @return the reward
        """
        # If the object crossed the bottom line, then:
        # compute the reward, generate a new image and return true.
        if self.state[1] < 0.5:
            return self.compute_square_reward()
        else:
            return self.compute_non_square_reward()

    #
    # Getter and setter.
    #

    @property
    def y_pos(self):
        """!
        Getter.
        @return the current position of the object on the y-axis
        """
        return self.state[5]

    @y_pos.setter
    def y_pos(self, new_value):
        """!
        Setter.
        @param new_value: the new position of the object on the y-axis
        """
        self.state[5] = new_value

    @property
    def x_pos(self):
        """!
        Getter.
        @return the current position of the object on the x-axis
        """
        return self.state[4]

    @x_pos.setter
    def x_pos(self, new_value):
        """!
        Setter.
        @param new_value: the new position of the object on the x-axis
        """
        self.state[4] = new_value
