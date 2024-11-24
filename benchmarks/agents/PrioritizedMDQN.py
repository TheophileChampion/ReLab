from benchmarks.agents.DQN import ReplayType, NetworkType, LossType, DQN


class PrioritizedMDQN(DQN):
    """
    Implement the multistep [1] Deep Q-Network agent [2] with prioritized replay buffer [3] from:

    [1] Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3:9–44, 1988.
    [2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves,
        Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.
        Human-level control through deep reinforcement learning. nature, 2015.
    [3] Tom Schaul. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
    """

    def __init__(
        self, gamma=0.99, learning_rate=0.00001, buffer_size=1000000, batch_size=32, learning_starts=200000, kappa=None,
        target_update_interval=40000, adam_eps=1.5e-4, n_actions=18, n_atoms=1, v_min=None, v_max=None, training=True,
        n_steps=3, replay_type=ReplayType.MULTISTEP_PRIORITIZED, loss_type=LossType.DQN_SL1,
        network_type=NetworkType.DEFAULT
    ):
        """
        Create a DQN agent.
        :param gamma: the discount factor
        :param learning_rate: the learning rate
        :param buffer_size: the size of the replay buffer
        :param batch_size: the size of the batches sampled from the replay buffer
        :param learning_starts: the step at which learning starts
        :param kappa: the kappa parameter of the quantile Huber loss see Equation (10) in QR-DQN paper
        :param target_update_interval: number of training steps between two synchronization of the target
        :param adam_eps: the epsilon parameter of the Adam optimizer
        :param n_actions: the number of actions available to the agent
        :param n_atoms: the number of atoms used to approximate the distribution over returns
        :param v_min: the minimum amount of returns (only used for categorical DQN)
        :param v_max: the maximum amount of returns (only used for categorical DQN)
        :param training: True if the agent is being trained, False otherwise
        :param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        :param replay_type: the type of replay buffer
        :param loss_type: the loss to use during gradient descent
        :param network_type: the network architecture to use for the value and target networks
        """

        # Call the parent constructor.
        super().__init__(
            gamma=gamma, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size,
            learning_starts=learning_starts, kappa=kappa, target_update_interval=target_update_interval,
            adam_eps=adam_eps, n_actions=n_actions, n_atoms=n_atoms, v_min=v_min, v_max=v_max, training=training,
            n_steps=n_steps, replay_type=replay_type, loss_type=loss_type, network_type=network_type
        )
