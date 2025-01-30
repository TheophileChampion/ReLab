import os
from math import prod
from typing import Union, Optional

from torch import zeros_like
from torch.optim import Adam
from torch import nn
import gc
from matplotlib.widgets import Button

from matplotlib import colors
import matplotlib as mpl
import matplotlib.pyplot as plt

from torch.distributions.multivariate_normal import MultivariateNormal
from torch import zeros, eye

import math
from os import listdir, makedirs
from os.path import isfile, isdir, join, exists, dirname
from pathlib import Path
import importlib
import torch
import re
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import abc
from abc import ABC
import logging
from collections import deque

import collections
import numpy as np
from torch import cat, FloatTensor, BoolTensor, IntTensor, unsqueeze

from relab.agents.networks.DecoderNetworks import ContinuousDecoderNetwork
from relab.agents.networks.EncoderNetworks import ContinuousEncoderNetwork


class SelectRandomAction:
    """
    Class that performs a random action selection.
    """

    def __init__(self, **_):
        pass

    def __iter__(self):
        """
        Make the class iterable.
        :return: the next key and value.
        """
        for key, value in {
            "module": str(self.__module__),
            "class": str(self.__class__.__name__)
        }.items():
            yield key, value

    def select(self, quality, steps_done):
        """
        Select a random action.
        :param quality: a vector containing the quality of all actions (unused).
        :param steps_done: the number of steps performed in the environment to date.
        :return: the selected action.
        """
        return np.random.choice(quality.shape[1])


#
# Class storing an experience.
#
Experience = collections.namedtuple(
    'Experience',
    field_names=[
        'obs',
        'action',
        'reward',
        'done',
        'next_obs'])


class ReplayBuffer:
    """
    Class implementing the experience replay buffer.
    """

    def __init__(self, capacity=10000):
        """
        Constructor
        :param capacity: the number of experience the buffer can store
        """
        self.buffer = collections.deque(maxlen=capacity)
        self.device = Device.get()

    def __len__(self):
        """
        Getter
        :return: the number of elements contained in the replay buffer
        """
        return len(self.buffer)

    def append(self, experience):
        """
        Add a new experience to the buffer
        :param experience: the experience to add
        """
        self.buffer.append(experience)

    @staticmethod
    def list_to_tensor(tensor_list):
        """
        Transform a list of n dimensional tensors into a tensor with n+1 dimensions
        :param tensor_list: the list of tensors
        :return: the output tensor
        """
        return cat([unsqueeze(tensor, 0) for tensor in tensor_list])

    def sample(self, batch_size):
        """
        Sample a batch from the replay buffer
        :param batch_size: the size of the batch to sample
        :return: observations, actions, rewards, done, next_observations
        where:
        - observations: the batch of observations
        - actions: the actions performed
        - rewards: the rewards received
        - done: whether the environment stop after performing the actions
        - next_observations: the observations received after performing the actions
        """

        # Sample a batch from the replay buffer.
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, actions, rewards, done, next_obs = zip(
            *[self.buffer[idx] for idx in indices])

        # Convert the batch into a torch tensor stored on the proper device.
        return self.list_to_tensor(obs).to(self.device), \
            IntTensor(actions).to(self.device), \
            FloatTensor(rewards).to(self.device), \
            BoolTensor(done).to(self.device), \
            self.list_to_tensor(next_obs).to(self.device)


def safe_mean(arr: Union[np.ndarray, list, deque]) -> float:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    return np.nan if len(arr) == 0 else float(
        np.mean(arr))  # type: ignore[arg-type]


class AgentInterface(ABC):
    """
    The interface that all agents need to implement.
    """

    def __init__(self, tensorboard_dir, steps_done=1, need_writer=True):
        """
        Construct an agent
        :param tensorboard_dir: the directory in which the tensorboard logs should be written
        :param steps_done: the number of training steps done so far
        :param need_writer: if true create a SummaryWriter
        """

        # Create the queue containing the episode information.
        self.ep_info_buffer = deque(maxlen=100)

        # Create the summary writer for monitoring with TensorBoard.
        self.writer = SummaryWriter(tensorboard_dir) if need_writer else None

        # Number of training steps performed to date.
        self.steps_done = steps_done

    @abc.abstractmethod
    def step(self, obs):
        """
        Select the next action to perform in the environment
        :param obs: the observation available to make the decision
        :return: the next action to perform
        """
        ...

    @abc.abstractmethod
    def name(self):
        """
        Getter
        :return: the agent's name
        """
        ...

    @abc.abstractmethod
    def n_steps_done(self):
        """
        Getter
        :return: the number of training steps performed to date
        """
        ...

    @abc.abstractmethod
    def total_rewards_obtained(self):
        """
        Getter
        :return: the total number of rewards gathered to date
        """
        ...

    @abc.abstractmethod
    def train(self, env):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        """
        ...

    def test(
            self,
            env,
            config,
            reward_name=None,
            n_steps_done=0,
            total_rewards=0):
        """
        Test the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        :param reward_name: the reward name as displayed in tensorboard
        :param n_steps_done: the number of steps already performed in the environment
        :param total_rewards: the total amount of rewards obtained to date
        """

        # Retrieve the initial observation from the environment.
        obs = env.reset()

        # Test the agent.
        task_performed = "training" if reward_name is None else "testing"
        logging.info(f"Start the {task_performed} at {datetime.now()}")
        while n_steps_done < config.task.max_n_steps:

            # Select an action.
            action = self.step(obs)

            # Execute the action in the environment.
            obs, reward, done, info = env.step(action)

            # Monitor total reward if needed.
            if self.writer is not None:
                total_rewards += reward
                if self.steps_done % config.tensorboard.log_interval == 0:
                    self.writer.add_scalar(
                        "total_rewards", total_rewards, self.steps_done)
                    self.log_episode_info(info, config.task.name)

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()

            # Increase the number of iterations performed.
            n_steps_done += 1

        # Close the environment.
        env.close()

    def compute_mean_episodic_reward(self):
        """
        Compute the mean episodic reward
        :return: the mean episodic reward, if it can be computed, None otherwise
        """
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            return safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
        return None

    def store_episode_info(self, information):
        """
        Store the episode information into the internal queue
        :param information: the information to store
        """

        # Make sure that the information is stored as a list of dictionary.
        if not isinstance(information, list):
            information = [information]

        # Store episode information in the internal queue.
        for info in information:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.ep_info_buffer.extend([ep_info])

    def log_episode_info(self, information, task_name, steps_done=-1):
        """
        Log episode information in tensorboard
        :param information: the information returned by the environment
        :param task_name: the name of the task being performed
        :param steps_done: the number of steps done so far
        """

        # Make sure that the number of steps done is valid.
        steps_done = steps_done if steps_done >= 0 else self.steps_done

        # Store episode information.
        self.store_episode_info(information)

        # Log mean episodic reward and mean episode length.
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:

            # Log mean episodic reward into tensorboard, and report it to ray
            # tune (if needed).
            ep_rew_mean = self.compute_mean_episodic_reward()
            self.writer.add_scalar(
                "rollout/ep_rew_mean", ep_rew_mean, steps_done)

            # Log mean episode length.
            ep_len_mean = safe_mean([ep_info["l"]
                                    for ep_info in self.ep_info_buffer])
            self.writer.add_scalar(
                "rollout/ep_len_mean", ep_len_mean, steps_done)

    def draw_reconstructed_images(self, env, grid_size):
        """
        Get reconstructed images
        :param env: the gym environment
        :param grid_size: the size of the image grid to generate
        """
        raise Exception(
            "The function 'get_reconstructed_images' is not implemented by this agent.")


class Checkpoint:
    """
    Class allowing the loading of model checkpoints.
    """

    def __init__(self, tb_dir, directory):
        """
        Construct the checkpoint from the checkpoint file
        :param tb_dir: the path of tensorboard directory
        :param directory: the checkpoint directory
        """

        # Get the latest checkpoint file
        checkpoint_file = self.get_latest_checkpoint(directory)
        if checkpoint_file is None:
            self.checkpoint = None
            return

        # Load checkpoint from path.
        self.checkpoint = torch.load(
            checkpoint_file, map_location=Device.get())
        if "checkpoint_dir" in self.checkpoint and self.checkpoint["checkpoint_dir"] != directory:
            self.checkpoint["checkpoint_dir"] = directory
            logging.info(
                "The given checkpoint directory does not match the one found in the file.")
            logging.info("Overwriting checkpoint directory to: " + directory)

        # Store the tensorboard directory
        self.tb_dir = tb_dir
        if "tensorboard_dir" in self.checkpoint and self.checkpoint["tensorboard_dir"] != directory:
            self.checkpoint["tensorboard_dir"] = directory
            logging.info(
                "The given tensorboard directory does not match the one found in the file.")
            logging.info("Overwriting tensorboard directory to: " + directory)

    @staticmethod
    def get_latest_checkpoint(directory, regex=r'model_\d+.pt'):
        """
        Get the latest checkpoint file matching the regex
        :param directory: the checkpoint directory
        :param regex: the regex checking whether a file name is a valid checkpoint file
        :return: None if an error occurred, else the path to the latest checkpoint
        """
        # If the path is not a directory or does not exist, return without
        # trying to load the checkpoint.
        if not exists(directory) or not isdir(directory):
            logging.warning(
                "The following directory was not found: " +
                directory)
            return None

        # If the directory does not contain any files, return without trying to
        # load the checkpoint.
        files = [
            file for file in listdir(directory) if isfile(
                join(
                    directory,
                    file))]
        if len(files) == 0:
            logging.warning("No checkpoint found in directory: " + directory)
            return None

        # Retrieve the file whose name contain the largest number.
        # This number is assumed to be the time step at which the agent was
        # saved.
        max_number = - math.inf
        file = None
        for curr_file in files:
            # Retrieve the number of training steps of the current checkpoint
            # file.
            if len(re.findall(regex, curr_file)) == 0:
                continue
            current_number = max([int(number)
                                 for number in re.findall(r'\d+', curr_file)])

            # Remember the checkpoint file with the highest number of training
            # steps.
            if current_number > max_number:
                max_number = current_number
                file = join(directory, curr_file)

        logging.info("Loading agent from the following file: " + file)
        return file

    def exists(self):
        """
        Check whether the checkpoint file exists.
        :return: True if the checkpoint file exists, False otherwise.
        """
        return self.checkpoint is not None

    def load_agent(self, training_mode=True, override=None):
        """
        Load the agent from the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :param override: the key-value pairs that need to be overridden in the checkpoint
        :return: the loaded agent or None if an error occurred.
        """

        # Check if the checkpoint is loadable.
        if not self.exists():
            return None

        # Override key-value pairs in the checkpoint if needed.
        if override is not None:
            for key, value in override.items():
                self.checkpoint[key] = value

        # Load the agent class and module.
        agent_module = importlib.import_module(self.checkpoint["agent_module"])
        agent_class = getattr(agent_module, self.checkpoint["agent_class"])

        # Load the parameters of the constructor from the checkpoint.
        param = agent_class.load_constructor_parameters(
            self.tb_dir, self.checkpoint, training_mode)

        # Instantiate the agent.
        return agent_class(**param)

    @staticmethod
    def create_dir_and_file(checkpoint_file):
        """
        Create the directory and file of the checkpoint if they do not already exist
        :param checkpoint_file: the checkpoint file
        """
        checkpoint_dir = dirname(checkpoint_file)
        if not exists(checkpoint_dir):
            makedirs(checkpoint_dir)
            file = Path(checkpoint_file)
            file.touch(exist_ok=True)

    @staticmethod
    def set_training_mode(neural_net, training_mode):
        """
        Set the training mode of the neural network sent as parameters
        :param neural_net: the neural network whose training mode needs to be set
        :param training_mode: True if the agent is being loaded for training, False otherwise
        """
        if training_mode:
            neural_net.train()
        else:
            neural_net.eval()

    @staticmethod
    def load_value_network(checkpoint, training_mode, prefix=""):
        """
        Load the value network from the checkpoint passed as parameters
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :param prefix: the value network prefix
        :return: the value network
        """

        # Load value network.
        value_net_module = importlib.import_module(
            checkpoint[f"value_net{prefix}_module"])
        value_net_class = getattr(value_net_module,
                                  checkpoint[f"value_net{prefix}_class"])
        value_net = value_net_class(
            n_actions=checkpoint["n_actions"],
            n_states=checkpoint["n_states"])
        value_net.load_state_dict(checkpoint[f"value_net{prefix}_state_dict"])

        # Set the training mode of the value network.
        Checkpoint.set_training_mode(value_net, training_mode)
        return value_net

    @staticmethod
    def load_value_networks(checkpoint, training_mode, n_value_networks):
        """
        Load the value networks from the checkpoint passed as parameters
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :param n_value_networks: the number of value networks to load
        :return: the value networks
        """
        return [
            Checkpoint.load_value_network(checkpoint, training_mode, prefix=f"_{i}")
            for i in range(n_value_networks)
        ]

    @staticmethod
    def load_decoder(checkpoint, training_mode=True):
        """
        Load the decoder from the checkpoint
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: the decoder
        """

        # Load number of states and the image shape.
        # image_shape = checkpoint["image_shape"]
        n_states = checkpoint["n_states"]

        # Load decoder network.
        decoder_module = importlib.import_module(
            checkpoint["decoder_net_module"])
        decoder_class = getattr(
            decoder_module,
            checkpoint["decoder_net_class"])
        decoder = decoder_class(n_states, 4)
        decoder.load_state_dict(checkpoint["decoder_net_state_dict"])

        # Set the training mode of the decoder.
        Checkpoint.set_training_mode(decoder, training_mode)
        return decoder

    @staticmethod
    def load_encoder(checkpoint, training_mode=True):
        """
        Load the encoder from the checkpoint
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: the encoder
        """

        # Load number of states and the image shape.
        # image_shape = checkpoint["image_shape"]
        n_states = checkpoint["n_states"]
        # n_actions = checkpoint["n_actions"] if "n_actions" in checkpoint else -1

        # Load encoder network.
        encoder_module = importlib.import_module(
            checkpoint["encoder_net_module"])
        encoder_class = getattr(
            encoder_module,
            checkpoint["encoder_net_class"])
        encoder = encoder_class(n_states, 4)
        encoder.load_state_dict(checkpoint["encoder_net_state_dict"])

        # Set the training mode of the encoder.
        Checkpoint.set_training_mode(encoder, training_mode)
        return encoder

    @staticmethod
    def load_transition(checkpoint, training_mode=True):
        """
        Load the transition from the checkpoint
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: the transition
        """

        # Load transition network.
        transition_module = importlib.import_module(
            checkpoint["transition_net_module"])
        transition_class = getattr(
            transition_module,
            checkpoint["transition_net_class"])
        transition = transition_class(
            n_states=checkpoint["n_states"], n_actions=checkpoint["n_actions"]
        )
        transition.load_state_dict(checkpoint["transition_net_state_dict"])

        # Set the training mode of the transition.
        Checkpoint.set_training_mode(transition, training_mode)
        return transition

    @staticmethod
    def load_critic(
            checkpoint,
            training_mode=True,
            n_states_key="n_states",
            network_key="critic_net"):
        """
        Load the critic from the checkpoint
        :param checkpoint: the checkpoint
        :param n_states_key: the key of the dictionary containing the number of states
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :param network_key: the prefix of the keys containing the critic's module and class
        :return: the critic.
        """
        # Check validity of inputs
        if network_key + '_module' not in checkpoint.keys() or network_key + \
                '_class' not in checkpoint.keys():
            return None

        # Load critic network.
        critic_module = importlib.import_module(
            checkpoint[network_key + "_module"])
        critic_class = getattr(critic_module,
                               checkpoint[network_key + "_class"])
        image_shape = checkpoint["image_shape"] if "image_shape" in checkpoint else None
        critic = critic_class(
            n_states=checkpoint[n_states_key],
            n_actions=checkpoint["n_actions"],
            image_shape=image_shape)
        critic.load_state_dict(checkpoint[network_key + "_state_dict"])

        # Set the training mode of the critic.
        Checkpoint.set_training_mode(critic, training_mode)
        return critic

    @staticmethod
    def load_value(
            checkpoint,
            training_mode=True,
            n_states_key="n_states",
            network_key="value_net"):
        """
        Load the value from the checkpoint
        :param checkpoint: the checkpoint
        :param n_states_key: the key of the dictionary containing the number of states
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :param network_key: the prefix of the keys containing the value's module and class
        :return: the value.
        """
        # Check validity of inputs
        if network_key + '_module' not in checkpoint.keys() or network_key + \
                '_class' not in checkpoint.keys():
            return None

        # Load value network.
        value_module = importlib.import_module(
            checkpoint[network_key + "_module"])
        value_class = getattr(value_module, checkpoint[network_key + "_class"])
        value = value_class(
            n_states=checkpoint[n_states_key],
            n_actions=checkpoint["n_actions"])
        value.load_state_dict(checkpoint[network_key + "_state_dict"])

        # Set the training mode of the value.
        Checkpoint.set_training_mode(value, training_mode)
        return value

    @staticmethod
    def load_object_from_dictionary(checkpoint, key):
        """
        Load an object from the checkpoint passed as parameters
        :param checkpoint: the checkpoint
        :param key: the key in the dictionary where the object has been serialized
        :return: the loaded object
        """

        # Load the object from the checkpoint.
        obj = checkpoint[key]
        obj_module = importlib.import_module(obj["module"])
        obj_class = getattr(obj_module, obj["class"])
        return obj_class(**obj)


def entropy_gaussian(log_var, sum_dims=None):
    """
    Compute the entropy of a Gaussian distribution
    :param log_var: the logarithm of the variance parameter
    :param sum_dims: the dimensions along which to sum over before to return, by default only dimension one
    :return: the entropy of a Gaussian distribution
    """
    ln2pie = 1.23247435026
    sum_dims = [1] if sum_dims is None else sum_dims
    return log_var.size()[1] * 0.5 * ln2pie + 0.5 * log_var.sum(sum_dims)


def kl_div_categorical(pi_hat, pi, epsilon=10e-5):
    """
    Compute the KL-divergence between two categorical distribution.
    :param pi_hat: the parameters of the first categorical distribution.
    :param pi: the parameters of the second categorical distribution.
    :param epsilon: a small value added to the probabilities to avoid taking the logarithm of zero
    :return: the KL-divergence.
    """
    kl = pi_hat * ((pi_hat + epsilon).log() - (pi + epsilon).log())
    return kl.sum()


def kl_div_categorical_with_logits(log_pi_hat, log_pi):
    """
    Compute the KL-divergence between two categorical distribution.
    :param log_pi_hat: the logarithm of the parameters of the first categorical distribution.
    :param log_pi: the logarithm of the parameters of the second categorical distribution.
    :return: the KL-divergence.
    """
    kl = torch.softmax(log_pi_hat, dim=1) * (log_pi_hat - log_pi)
    return kl.sum()


def kl_div_gaussian(
        mean_hat,
        log_var_hat,
        mean=None,
        log_var=None,
        sum_dims=None,
        min_var=10e-4):
    """
    Compute the KL-divergence between two Gaussian distributions
    :param mean_hat: the mean of the first Gaussian distribution
    :param log_var_hat: the logarithm of variance of the first Gaussian distribution
    :param mean: the mean of the second Gaussian distribution
    :param log_var: the logarithm of variance of the second Gaussian distribution
    :param sum_dims: the dimensions along which to sum over before to return, by default all of them
    :param min_var: the minimal variance allowed to avoid division by zero
    :return: the KL-divergence between the two Gaussian distributions
    """

    # Initialise the mean and log variance vectors to zero, if they are not
    # provided as parameters.
    if mean is None:
        mean = torch.zeros_like(mean_hat)
    if log_var is None:
        log_var = torch.zeros_like(log_var_hat)

    # Compute the KL-divergence
    var = log_var.exp()
    var = torch.clamp(var, min=min_var)
    kl_div = log_var - log_var_hat + \
        torch.exp(log_var_hat - log_var) + (mean - mean_hat) ** 2 / var

    if sum_dims is None:
        return 0.5 * kl_div.sum(dim=1).mean()
    else:
        return 0.5 * kl_div.sum(dim=sum_dims)


def log_bernoulli_with_logits(obs, alpha):
    """
    Compute the log probability of the observation (obs), given the logits (alpha), assuming
    a bernoulli distribution, c.f.
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    :param obs: the observation
    :param alpha: the logits
    :return: the log probability of the observation
    """
    one = torch.ones_like(alpha)
    zero = torch.zeros_like(alpha)
    out = - torch.maximum(alpha, zero) + alpha * obs - \
        torch.log(one + torch.exp(-torch.abs(alpha)))
    return out.sum(dim=(1, 2, 3)).mean()


def reparameterize(mean, log_var):
    """
    Perform the reparameterization trick
    :param mean: the mean of the Gaussian
    :param log_var: the log of the variance of the Gaussian
    :return: a sample from the Gaussian on which back-propagation can be performed
    """
    nb_states = mean.shape[1]
    epsilon = MultivariateNormal(zeros(nb_states), eye(
        nb_states)).sample([mean.shape[0]]).to(Device.get())
    return epsilon * torch.exp(0.5 * log_var) + mean


def compute_info_gain(g_value, mean, log_var, mean_hat, log_var_hat):
    """
    Compute the information gain.
    :param g_value: the definition of the efe to use, i.e., reward or efe.
    :param mean_hat: the mean from the encoder.
    :param log_var_hat: the log variance from the encoder.
    :param mean: the mean from the transition.
    :param log_var: the log variance from the transition.
    :return: the information gain.
    """
    info_gain = torch.zeros([1]).to(Device.get())
    if g_value == "old_efe":
        info_gain = -kl_div_gaussian(mean, log_var, mean_hat, log_var_hat)
    if g_value == "efe":
        info_gain = kl_div_gaussian(mean_hat, log_var_hat, mean, log_var)
    if g_value == "entropy_posterior":
        info_gain = - entropy_gaussian(log_var_hat)
    if g_value == "entropy_prior":
        info_gain = - entropy_gaussian(log_var)
    if g_value == "entropy":
        info_gain = - entropy_gaussian(log_var) - entropy_gaussian(log_var_hat)
    return info_gain


class PlotsBuilder:
    """
    Allows the creation of complex plots to visualize models based on Gaussian Mixture.
    """

    def __init__(self, title, n_rows=1, n_cols=1):

        # Store number of rows and columns.
        self.n_cols = n_cols
        self.n_rows = n_rows

        # Create the subplots, and set the main title.
        self.f, self.axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        self.f.suptitle(title)

        # Create the list of colors to use.
        self.colors = list(colors.CSS4_COLORS.keys())
        first_colors = [
            'red',
            'green',
            'blue',
            'purple',
            'gray',
            'pink',
            'turquoise',
            'orange',
            'brown',
            'cyan']
        for i, color in enumerate(first_colors):
            index = self.colors.index(color)
            i_color = self.colors[i]
            self.colors[i] = color
            self.colors[index] = i_color

        # Index of the current plot.
        self.current_plot_index = 0

    @property
    def current_axis(self):
        if self.n_rows == 1 and self.n_cols == 1:
            return self.axes
        if self.n_rows == 1 or self.n_cols == 1:
            return self.axes[self.current_plot_index]
        return self.axes[int(self.current_plot_index /
                             self.n_cols)][self.current_plot_index %
                                           self.n_cols]

    def draw_gaussian_mixture(
            self,
            title="",
            data=None,
            r=None,
            params=None,
            clusters=False,
            ellipses=True):

        # Set the subplot title.
        self.current_axis.set_title(title)

        # Draw the data points.
        if data is not None:
            # Draw the data points of t = 0.
            x = [x_tensor[0] for x_tensor in data]
            y = [x_tensor[1] for x_tensor in data]

            r = torch.softmax(r, dim=1)
            c = [tuple(r_n) for r_n in r] if r.shape[1] == 3 else [
                self.colors[torch.argmax(r_n)] for r_n in r]
            self.current_axis.scatter(x=x, y=y, c=c)

        # Draw the ellipses corresponding to the current model believes.
        if ellipses is True:
            active_components = set(r.argmax(dim=1).tolist())
            self.make_ellipses(active_components, params)

        # Draw the cluster center.
        if clusters is True:
            μ, _, _ = params
            x = [μ_k[0] for μ_k in μ]
            y = [μ_k[1] for μ_k in μ]
            self.current_axis.scatter(
                x=x, y=y, marker="X", s=100, c="black", edgecolor="white")

        # Move to the next axis.
        self.current_plot_index += 1

    def make_ellipses(self, active_components, params):
        m_hat, v_hat, W_hat = params
        for k in range(len(v_hat)):
            if k not in active_components:
                continue
            color = self.colors[k]

            covariances = torch.inverse(v_hat[k] * W_hat[k])
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi
            v = 3. * np.sqrt(2.) * np.sqrt(np.maximum(v, 0))
            mean = m_hat[k]
            mean = mean.reshape(2, 1)
            ell = mpl.patches.Ellipse(
                mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(self.current_axis.bbox)
            ell.set_alpha(0.5)
            self.current_axis.add_artist(ell)
            self.current_axis.set_aspect('equal', 'datalim')

    def draw_matrix(self, matrix, title=""):

        # Set the subplot title.
        self.current_axis.set_title(title)

        # Draw the matrix passed as parameters.
        plt.sca(self.current_axis)
        axis_img = plt.matshow(matrix, fignum=0)
        plt.colorbar(axis_img)

        # Move to the next axis.
        self.current_plot_index += 1

    def draw_responsibility_histograms(self, r, title=""):

        # Set the subplot title.
        self.current_axis.set_title(title)

        # Retrieve the dataset size and the number of states.
        n_states = r.shape[1]

        # Draw a bar plot representing how many point are attributed to each
        # component.
        x = [state for state in range(n_states)]
        y = r.sum(dim=0).tolist()
        bars = self.current_axis.bar(x, y, align='center')
        for state in range(n_states):
            bars[state].set_color(self.colors[state])
            bars[state].set_alpha(0.53)

        # Move to the next axis.
        self.current_plot_index += 1

    @staticmethod
    def show():
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()


class MatPlotLib:
    """
    A helper class providing useful functions for interacting with matplotlib.
    """

    @staticmethod
    def close(fig=None):
        """
        Close the figure passed as parameter or the current figure
        :param fig: the figure to close
        """

        # Clear the current axes.
        plt.cla()

        # Clear the current figure.
        plt.clf()

        # Closes all the figure windows.
        plt.close('all')

        # Closes the matplotlib figure
        plt.close(plt.gcf() if fig is None else fig)

        # Forces the garbage collection
        gc.collect()

    @staticmethod
    def format_image(img):
        """
        Turn a 4d pytorch tensor into a 3d numpy array
        :param img: the 4d tensor
        :return: the 3d array
        """
        return torch.squeeze(img)[
            :3, :, :].swapaxes(
            0, 1).swapaxes(
            1, 2).detach().cpu().numpy()

    @staticmethod
    def save_figure(out_f_name, dpi=300, tight=True):
        """
        Save a matplotlib figure in an `out_f_name` file.
        :param str out_f_name: Name of the file used to save the figure.
        :param int dpi: Number of dpi, Default 300.
        :param bool tight: If True, use plt.tight_layout() before saving. Default True.
        """
        if tight is True:
            plt.tight_layout()
        plt.savefig(out_f_name, dpi=dpi, transparent=True)
        MatPlotLib.close()

    @staticmethod
    def draw_gm_graph(
            params,
            data,
            r,
            title="",
            clusters=False,
            ellipses=True,
            skip_fc=None):
        """
        Draw the Gaussian Mixture graph
        :param params: a 3-tuples of the form (m_hat, v_hat, W_hat)
        :param data: the data points
        :param r: the responsibilities for all data points
        :param title: the title of the figure
        :param clusters: whether to draw the cluster centers
        :param ellipses: whether to draw the ellipses
        :param skip_fc: the function to call when the skip button is clicked
        """

        # Draw the plots.
        plots = PlotsBuilder(title, n_cols=2)
        plots.draw_gaussian_mixture(
            title="Observation at t = 0",
            data=data,
            r=r,
            params=params,
            clusters=clusters,
            ellipses=ellipses)
        plots.draw_responsibility_histograms(
            title="Responsibilities at t = 0", r=r)

        # Add the skip button.
        if skip_fc is not None:
            plt.gcf().add_axes([0.97, 0.97, 0.02, 0.02])
            b_next = Button(
                plt.gca(),
                'Skip',
                color="limegreen",
                hovercolor="forestgreen")
            b_next.on_clicked(skip_fc)

        # Display the graph.
        plots.show()

    @staticmethod
    def draw_tgm_graph(
            action_names,
            params,
            x0,
            x1,
            a0,
            r,
            title="",
            clusters=False,
            ellipses=True):
        """
        Draw the Temporal Gaussian Mixture graph
        :param action_names: name of all the environment's actions
        :param params: a 3-tuples of the form (m_hat, v_hat, W_hat)
        :param a0: the actions at time step zero
        :param x0: the data points at time step zero
        :param x1: the data points at time step one
        :param r: the responsibilities for all data points at time steps zero and one
        :param title: the title of the figure
        :param clusters: whether to draw the cluster centers
        :param ellipses: whether to draw the ellipses
        """

        # Retrieve the number of actions.
        n_actions = len(action_names)

        # Create the plot builder.
        plots = PlotsBuilder(
            title,
            n_rows=1 +
            math.ceil(
                n_actions /
                4.0),
            n_cols=4)

        # Draw the model's beliefs.
        plots.draw_gaussian_mixture(
            title="Observation at t = 0",
            data=x0,
            r=r[0],
            params=params,
            clusters=clusters,
            ellipses=ellipses)
        plots.draw_gaussian_mixture(
            title="Observation at t = 1",
            data=x1,
            r=r[1],
            params=params,
            clusters=clusters,
            ellipses=ellipses)

        # Draw the responsibilities.
        plots.draw_responsibility_histograms(
            title="Responsibilities at t = 0", r=r[0])
        plots.draw_responsibility_histograms(
            title="Responsibilities at t = 1", r=r[1])

        # Show all the plots.
        plots.show()

    @staticmethod
    def draw_dirichlet_tmhgm_graph(
            action_names,
            params0,
            params1,
            x0,
            x1,
            a0,
            r,
            b_hat,
            title="",
            clusters=False,
            ellipses=True,
            skip_fc=None):
        """
        Draw the Temporal Gaussian Mixture graph
        :param action_names: name of all the environment's actions
        :param params0: a 3-tuples of the form (m_hat, v_hat, W_hat)
        :param params1: a 3-tuples of the form (m_hat, v_hat, W_hat)
        :param a0: the actions at time step zero
        :param x0: the data points at time step zero
        :param x1: the data points at time step one
        :param r: the responsibilities for all data points at time steps zero and one
        :param b_hat: the non-normalized counts of the transition
        :param title: the title of the figure
        :param clusters: whether to draw the cluster centers
        :param ellipses: whether to draw the ellipses
        :param skip_fc: the function to call when the skip button is clicked
        """

        # Retrieve the number of actions.
        n_actions = len(action_names)

        # Create the plot builder.
        plots = PlotsBuilder(
            title,
            n_rows=1 +
            2 *
            math.ceil(
                n_actions /
                4.0),
            n_cols=4)

        # Draw the model's beliefs.
        plots.draw_gaussian_mixture(
            title="Observation at t = 0",
            data=x0,
            r=r[0],
            params=params0,
            clusters=clusters,
            ellipses=ellipses)
        plots.draw_gaussian_mixture(
            title="Observation at t = 1",
            data=x1,
            r=r[1],
            params=params1,
            clusters=clusters,
            ellipses=ellipses)

        # Draw the responsibilities.
        plots.draw_responsibility_histograms(
            title="Responsibilities at t = 0", r=r[0])
        plots.draw_responsibility_histograms(
            title="Responsibilities at t = 1", r=r[1])

        # Draw the transition matrix for each action.
        for action in range(n_actions):
            plots.draw_matrix(
                b_hat[action],
                title=f"Transition matrix for action = {
                    action_names[action]}")

        # Add the skip button.
        if skip_fc is not None:
            plt.gcf().add_axes([0.97, 0.97, 0.02, 0.02])
            b_next = Button(
                plt.gca(),
                'Skip',
                color="limegreen",
                hovercolor="forestgreen")
            b_next.on_clicked(skip_fc)

        # Show all the plots.
        plots.show()


class Device:
    """
    Singleton to access the type of device to use, i.e. GPU or CPU.
    """

    # Specify the device on which models and tensors must be sent (i.e., None,
    # cpu or cuda).
    device_name = "cuda"

    @staticmethod
    def get():
        """
        Getter
        :return: the device on which computation should be performed
        """

        # If the device was not provided by the user:
        # - select cuda if cuda is available
        # - select cpu otherwise
        if Device.device_name is None:
            Device.device_name = 'cuda' if torch.cuda.is_available(
            ) and torch.cuda.device_count() >= 1 else 'cpu'

        # Create the device.
        return torch.device(Device.device_name)

    @staticmethod
    def send(models):
        """
        Send the models to the device, i.e., gpu or cpu
        :param models: the list of model to send to the device
        """
        device = Device.get()
        for model in models:
            model.to(device)


def get_adam(modules, lr):
    """
    Create and returns an adam optimizer.
    :param modules: the modules whose parameters must be optimizers.
    :param lr: the learning rate.
    :return: the adam optimizer.
    """
    params = []
    for module in modules:
        params += list(module.parameters())
    return Adam(params, lr=lr)

#
# Class implementing a network that maps a vector of size n into two vectors representing the mean
# and variance of a Gaussian with diagonal covariance matrix.
#


class DiagonalGaussian(nn.Module):

    def __init__(self, input_size, nb_components):
        """
        Constructor.
        :param input_size: size of the vector send as input of the layer.
        :param nb_components: the number of components of the diagonal Gaussian.
        """
        super().__init__()
        self.__mean = nn.Sequential(
            nn.Linear(input_size, nb_components)
        )
        self.__log_var = nn.Sequential(
            nn.Linear(input_size, nb_components),
        )

    def forward(self, x):
        """
        Compute the mean and the variance of the diagonal Gaussian (DG).
        :param x: the input vector
        :return: the mean and the log of the variance of the DG.
        """
        return self.__mean(x), self.__log_var(x)

#
# Class implementing a deconvolution decoder for 64 by 64 images.
#


class ConvDecoder64(nn.Module):

    def __init__(self, n_states, image_shape):
        """
        Constructor.
        :param n_states: the number of hidden variables, i.e., number of dimension in the Gaussian.
        :param image_shape: the shape of the input images.
        """

        super().__init__()

        # Create the deconvolutional network.
        self.__lin_net = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 1600),
            nn.ReLU(),
        )
        self.__up_conv_net = nn.Sequential(
            nn.ConvTranspose2d(
                64, 64, (4, 4), stride=(
                    2, 2), output_padding=(
                    1, 1)), nn.ReLU(), nn.ConvTranspose2d(
                64, 32, (4, 4), stride=(
                    2, 2), padding=(
                    0, 0), output_padding=(
                    1, 1)), nn.ReLU(), nn.ConvTranspose2d(
                32, 32, (4, 4), stride=(
                    2, 2), padding=(
                    0, 0), output_padding=(
                    1, 1)), nn.ReLU(), nn.ConvTranspose2d(
                32, image_shape[0], (4, 4), stride=(
                    1, 1), padding=(
                    0, 0), output_padding=(
                    0, 0)), )

    def forward(self, x):
        """
        Compute the shape parameters of a product of beta distribution.
        :param x: a hidden state.
        :return: the shape parameters of a product of beta distribution.
        """
        x = self.__lin_net(x)
        x = torch.reshape(x, (x.shape[0], 64, 5, 5))
        x = self.__up_conv_net(x)
        return x  # TODO .permute(0, 2, 3, 1)

#
# Class implementing a convolutional encoder for 64 by 64 images.
#


class ConvEncoder64(nn.Module):

    def __init__(self, n_states, image_shape, **_):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param image_shape: the shape of the input images.
        """

        super().__init__()

        # Create the convolutional encoder network.
        self.__conv_net = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, (4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (4, 4), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (2, 2), stride=(2, 2), padding=1),
            nn.ReLU(),
        )
        self.__conv_output_shape = self.__conv_output_shape(image_shape)
        self.__conv_output_shape = self.__conv_output_shape[1:]
        conv_output_size = prod(self.__conv_output_shape)

        # Create the linear encoder network.
        self.__linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            DiagonalGaussian(256, n_states)
        )

        # Create the full encoder network.
        self.__net = nn.Sequential(
            self.__conv_net,
            self.__linear_net
        )

    def __conv_output_shape(self, image_shape):
        """
        Compute the shape of the features output by the convolutional encoder.
        :param image_shape: the shape of the input image.
        :return: the shape of the features output by the convolutional encoder.
        """
        image_shape = list(image_shape)
        image_shape.insert(0, 1)
        input_image = zeros(image_shape)
        return self.__conv_net(input_image).shape

    def forward(self, x):
        """
        Forward pass through this encoder.
        :param x: the input.
        :return: the mean and logarithm of the variance of the Gaussian over latent variables.
        """
        # TODO x = x.permute(0, 3, 1, 2)
        return self.__net(x)


class VAE(AgentInterface):
    """
    Implement a Variational Auto-Encoder agent acting randomly.
    """

    def __init__(
            self,
            name="OldVAE",
            n_steps_beta_reset=10000000000,
            beta=1.0,
            lr=0.0001,
            beta_starting_step=0,
            beta_rate=0.0000,
            queue_capacity=50000,
            n_actions=18,
            n_states=10,
            image_shape=[
                4,
                64,
                64],
            steps_done=0,
            verbose=False,
            **_):
        """
        Constructor
        :param name: the agent name
        :param n_steps_beta_reset: the number of steps after with beta is reset
        :param beta_starting_step: the number of steps after which beta start increasing
        :param beta: the initial value for beta
        :param beta_rate: the rate at which the beta parameter is increased
        :param lr: the learning rate
        :param queue_capacity: the maximum capacity of the queue
        :param action_selection: the action selection to be used
        :param n_actions: the number of possible actions
        :param n_states: the number of latent states
        :param image_shape: the shape of the input image
        :param steps_done: the number of training iterations performed to date.
        :param verbose: whether to log weights information such as mean, min and max values of layers' weights
        """

        # Call parent constructor.
        super().__init__(os.environ["TENSORBOARD_DIRECTORY"], steps_done)

        # Neural networks.
        self.device = Device.get()
        # TODO self.encoder = ConvEncoder64(n_states, image_shape)
        self.encoder = ContinuousEncoderNetwork(n_states, 4)
        self.encoder.train(True)
        self.encoder.to(self.device)
        # TODO self.decoder = ConvDecoder64(n_states, image_shape)
        self.decoder = ContinuousDecoderNetwork(n_states, 4)
        self.decoder.train(True)
        self.decoder.to(self.device)

        # Ensure models are on the right device.
        Device.send([self.encoder, self.decoder])

        # Optimizer.
        self.optimizer = get_adam([self.encoder, self.decoder], lr)

        # Beta scheduling.
        self.n_steps_beta_reset = n_steps_beta_reset
        self.beta_starting_step = beta_starting_step
        self.beta = beta
        self.beta_rate = beta_rate

        # Miscellaneous.
        self.agent_name = name
        self.buffer = ReplayBuffer(capacity=queue_capacity)
        self.steps_done = steps_done
        self.lr = lr
        self.queue_capacity = queue_capacity
        self.tensorboard_dir = os.environ["TENSORBOARD_DIRECTORY"]
        self.checkpoint_dir = os.environ["CHECKPOINT_DIRECTORY"]
        self.action_selection = SelectRandomAction()
        self.n_actions = n_actions
        self.n_states = n_states
        self.image_shape = image_shape
        self.total_rewards = 0
        self.verbose = verbose

    def name(self):
        """
        Getter
        :return: the agent's name
        """
        return self.agent_name

    def n_steps_done(self):
        """
        Getter
        :return: the number of training steps performed to date
        """
        return self.steps_done

    def total_rewards_obtained(self):
        """
        Getter
        :return: the total number of rewards gathered to date
        """
        return self.total_rewards

    def step(self, obs):
        """
        Select a random action
        :param obs: unused
        :return: the action to be performed
        """
        quality = torch.zeros([1, self.n_actions]).to(Device.get())
        return self.action_selection.select(quality, self.steps_done)

    def train(self, env):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        :return: nothing
        """

        # Retrieve the initial observation from the environment.
        obs, _ = env.reset()

        # Train the agent.
        logging.info(f"Start the training at {datetime.now()}")
        while self.steps_done < 10000000:

            # Select an action.
            action = self.step(obs)

            # Execute the action in the environment.
            old_obs = obs
            obs, reward, done, trunc, info = env.step(action)
            done = done or trunc

            # Add the experience to the replay buffer.
            self.buffer.append(Experience(old_obs, action, reward, done, obs))

            # Perform one iteration of training (if needed).
            if len(self.buffer) >= 50000:
                self.learn()

            # Save the agent (if needed).
            if self.steps_done % 100000 == 0:
                self.save()

            # Log the reward (if needed).
            if self.writer is not None:
                self.total_rewards += reward
                if self.steps_done % 1 == 0:
                    self.writer.add_scalar(
                        "total_rewards", self.total_rewards, self.steps_done)
                    self.log_episode_info(info, "training")

            # Reset the environment when a trial ends.
            if done:
                obs, _ = env.reset()

            # Increase the number of steps done.
            self.steps_done += 1

        # Save the final version of the model.
        self.save(final_model=True)

        # Close the environment.
        env.close()

    def learn(self):
        """
        Perform on step of gradient descent on the encoder and the decoder
        :return: nothing
        """

        # Sample the replay buffer.
        _, _, _, _, next_obs = self.buffer.sample(50)

        # Compute the variational free energy.
        vfe_loss = self.compute_vfe(next_obs)
        if vfe_loss is None:
            return

        # Perform one step of gradient descent on the other networks.
        self.optimizer.zero_grad()
        vfe_loss.backward()
        self.optimizer.step()

        # Implement the cyclical scheduling for beta.
        if self.steps_done >= self.beta_starting_step:
            self.beta = np.clip(self.beta + self.beta_rate, 0, 1)
        if self.steps_done % self.n_steps_beta_reset == 0:
            self.beta = 0

    def compute_vfe(self, next_obs):
        """
        Compute the variational free energy
        :param next_obs: the observations at time t + 1
        :return: the variational free energy
        """

        # Compute required vectors.
        mean_hat, log_var_hat = self.encoder(next_obs)
        next_state = reparameterize(mean_hat, log_var_hat)
        mean = zeros_like(next_state)
        log_var = zeros_like(next_state)
        alpha = self.decoder(next_state)

        # Compute the variational free energy.
        kl_div_hs = kl_div_gaussian(mean_hat, log_var_hat, mean, log_var)
        log_likelihood = log_bernoulli_with_logits(next_obs, alpha)
        vfe_loss = self.beta * kl_div_hs - log_likelihood
        if torch.isnan(vfe_loss) or torch.isinf(vfe_loss) or vfe_loss > 1e5:
            return None

        # Display debug information, if needed.
        if self.writer is not None and self.steps_done % min(1, 50) == 0:

            # Log the mean, min and max values of weights, if requested by
            # user.
            if self.verbose and self.steps_done % min(1 * 10, 500) == 0:

                for neural_network in [self.encoder, self.decoder]:
                    for name, param in neural_network.named_parameters():
                        self.writer.add_scalar(
                            f"{name}.mean", param.mean(), self.steps_done)
                        self.writer.add_scalar(
                            f"{name}.min", param.min(), self.steps_done)
                        self.writer.add_scalar(
                            f"{name}.max", param.min(), self.steps_done)

            # Log the KL-divergence, the negative log likelihood, beta and the
            # variational free energy.
            self.writer.add_scalar("kl_div_hs", kl_div_hs, self.steps_done)
            self.writer.add_scalar(
                "neg_log_likelihood", - log_likelihood, self.steps_done)
            self.writer.add_scalar("beta", self.beta, self.steps_done)
            self.writer.add_scalar("vfe", vfe_loss, self.steps_done)

        return vfe_loss

    def predict(self, data):
        """
        Do one forward pass using given observation.
        :param data: a tuple containing the observations and actions at time t
        :return: the outputs of the encoder
        """
        obs, _ = data
        return self.encoder(obs)

    def save(self, final_model=False):
        """
        Create a checkpoint file allowing the agent to be reloaded later
        :param final_model: True if the model being saved is the final version, False otherwise
        """

        # Create directories and files if they do not exist.
        model_id = 10000000 if final_model is True else self.steps_done
        checkpoint_file = join(self.checkpoint_dir, f"model_{model_id}.pt")
        Checkpoint.create_dir_and_file(checkpoint_file)

        # Save the model.
        torch.save({
            "name": self.agent_name,
            "agent_module": str(self.__module__),
            "agent_class": str(self.__class__.__name__),
            "image_shape": self.image_shape,
            "n_states": self.n_states,
            "decoder_net_state_dict": self.decoder.state_dict(),
            "decoder_net_module": str(self.decoder.__module__),
            "decoder_net_class": str(self.decoder.__class__.__name__),
            "encoder_net_state_dict": self.encoder.state_dict(),
            "encoder_net_module": str(self.encoder.__module__),
            "encoder_net_class": str(self.encoder.__class__.__name__),
            "action_selection": dict(self.action_selection),
            "lr": self.lr,
            "beta": self.beta,
            "n_actions": self.n_actions,
            "n_steps_beta_reset": self.n_steps_beta_reset,
            "beta_starting_step": self.beta_starting_step,
            "beta_rate": self.beta_rate,
            "steps_done": self.steps_done,
            "queue_capacity": self.queue_capacity,
            "tensorboard_dir": self.tensorboard_dir,
            "checkpoint_dir": self.checkpoint_dir
        }, checkpoint_file)

    @staticmethod
    def load_constructor_parameters(tb_dir, checkpoint, training_mode=True):
        """
        Load the constructor parameters from a checkpoint.
        :param tb_dir: the path of tensorboard directory.
        :param checkpoint: the checkpoint from which to load the parameters.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: a dictionary containing the constructor's parameters.
        """
        return {
            "name": checkpoint["name"],
            "encoder": Checkpoint.load_encoder(
                checkpoint,
                training_mode),
            "decoder": Checkpoint.load_decoder(
                checkpoint,
                training_mode),
            "image_shape": checkpoint["image_shape"],
            "lr": checkpoint["lr"],
            "action_selection": Checkpoint.load_object_from_dictionary(
                checkpoint,
                "action_selection"),
            "beta": checkpoint["beta"],
            "n_actions": checkpoint["n_actions"],
            "n_states": checkpoint["n_states"],
            "n_steps_beta_reset": checkpoint["n_steps_beta_reset"],
            "beta_starting_step": checkpoint["beta_starting_step"],
            "beta_rate": checkpoint["beta_rate"],
            "steps_done": checkpoint["steps_done"],
            "queue_capacity": checkpoint["queue_capacity"],
            "tensorboard_dir": tb_dir,
            "checkpoint_dir": checkpoint["checkpoint_dir"],
        }

    @staticmethod
    def get_latest_checkpoint(regex: str = r"model_\d+.pt") -> Optional[str]:
        """!
        Get the latest checkpoint file matching the regex.
        @param regex: the regex checking whether a file name is a valid checkpoint file
        @return None if an error occurred, else the path to the latest checkpoint
        """

        # If the path is not a directory or does not exist, return without
        # trying to load the checkpoint.
        directory = os.environ["CHECKPOINT_DIRECTORY"]
        if not exists(directory) or not isdir(directory):
            logging.warning(
                "The following directory was not found: " +
                directory)
            return None

        # If the directory does not contain any files, return without trying to
        # load the checkpoint.
        files = [file for file in os.listdir(
            directory) if isfile(join(directory, file))]
        if len(files) == 0:
            logging.warning("No checkpoint found in directory: " + directory)
            return None

        # Retrieve the file whose name contain the largest number.
        # This number is assumed to be the time step at which the agent was
        # saved.
        max_number = - math.inf
        file = None
        for current_file in files:

            # Retrieve the number of training steps of the current checkpoint
            # file.
            if len(re.findall(regex, current_file)) == 0:
                continue
            current_number = max([int(number)
                                 for number in re.findall(r"\d+", current_file)])

            # Remember the checkpoint file with the highest number of training
            # steps.
            if current_number > max_number:
                max_number = current_number
                file = join(directory, current_file)

        return file

    def load(self, checkpoint_name: str = "",
             buffer_checkpoint_name: str = ""):
        """
        Create the agent according to the configuration
        :return: the created agent
        """

        # Retrieve the full agent checkpoint path.
        if checkpoint_name == "":
            checkpoint_path = self.get_latest_checkpoint()
        else:
            checkpoint_path = join(
                os.environ["CHECKPOINT_DIRECTORY"],
                checkpoint_name)

        # Check if the checkpoint can be loaded.
        if checkpoint_path is None:
            logging.info("Could not load the agent from the file system.")
            return

        # Load the checkpoint from the file system.
        logging.info(
            "Loading agent from the following file: " +
            checkpoint_path)
        self.device = Device.get()
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False)

        self.name = checkpoint["name"]
        self.encoder = Checkpoint.load_encoder(checkpoint, True)
        self.encoder.to(self.device)
        self.decoder = Checkpoint.load_decoder(checkpoint, True)
        self.decoder.to(self.device)
        self.image_shape = checkpoint["image_shape"]
        self.lr = checkpoint["lr"]
        self.action_selection = Checkpoint.load_object_from_dictionary(
            checkpoint, "action_selection")
        self.beta = checkpoint["beta"]
        self.n_actions = checkpoint["n_actions"]
        self.n_states = checkpoint["n_states"]
        self.n_steps_beta_reset = checkpoint["n_steps_beta_reset"]
        self.beta_starting_step = checkpoint["beta_starting_step"]
        self.beta_rate = checkpoint["beta_rate"]
        self.steps_done = checkpoint["steps_done"]
        self.queue_capacity = checkpoint["queue_capacity"]
        self.checkpoint_dir = checkpoint["checkpoint_dir"]

    def demo(self, env, gif_name, grid_size=(6, 6)):
        """
        Draw the ground truth and reconstructed images
        :param env: the gym environment
        :param grid_size: the size of the image grid to generate
        :return: the figure containing the images
        """

        # Create the figure and the grid specification.
        height, width = grid_size
        n_cols = 2
        fig = plt.figure(figsize=(width + n_cols, height * 2))
        gs = fig.add_gridspec(height * 2, width + n_cols)

        # Iterate over the grid's rows.
        for h in range(height):

            # Draw the ground truth label for each row.
            fig.add_subplot(gs[2 * h, 0:3])
            plt.text(0.08, 0.45, "Ground Truth Image:", fontsize=10)
            plt.axis('off')

            # Draw the reconstructed image label for each row.
            fig.add_subplot(gs[2 * h + 1, 0:3])
            plt.text(0.08, 0.45, "Reconstructed Image:", fontsize=10)
            plt.axis('off')

            # Iterate over the grid's columns.
            for w in range(width):

                # Retrieve the ground truth and reconstructed images.
                obs, _ = env.reset()
                obs = torch.unsqueeze(obs, dim=0).to(self.device)
                state, _ = self.encoder(obs)
                reconstructed_obs = torch.sigmoid(self.decoder(state))

                # Draw the ground truth image.
                fig.add_subplot(gs[2 * h, w + n_cols])
                plt.imshow(MatPlotLib.format_image(obs))
                plt.axis('off')

                # Draw the reconstructed image.
                fig.add_subplot(gs[2 * h + 1, w + n_cols])
                plt.imshow(MatPlotLib.format_image(reconstructed_obs))
                plt.axis('off')

        # Set spacing between subplots.
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=0.1)
        fig.savefig("./data/OldVAE.png")
        return fig
