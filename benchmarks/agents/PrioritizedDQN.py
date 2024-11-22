from benchmarks.agents.DQN import DQN, LossType, ReplayType, NetworkType


class PrioritizedDQN(DQN):
    """
    Implement the Deep Q-Network agent [1] with prioritized replay buffer [2] from:

    [1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves,
        Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.
        Human-level control through deep reinforcement learning. nature, 2015.
    [2] Tom Schaul. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
    """

    def __init__(
        self, gamma=0.99, learning_rate=0.00001, buffer_size=1000000, batch_size=32, learning_starts=200000,
        target_update_interval=40000, adam_eps=1.5e-4, replay_type=ReplayType.PRIORITIZED, loss_type=LossType.DQN_SL1,
        network_type=NetworkType.DEFAULT, training=True
    ):
        """
        Create a DQN agent.
        :param gamma: the discount factor
        :param learning_rate: the learning rate
        :param buffer_size: the size of the replay buffer
        :param batch_size: the size of the batches sampled from the replay buffer
        :param learning_starts: the step at which learning starts
        :param target_update_interval: number of training steps between two synchronization of the target
        :param adam_eps: the epsilon parameter of the Adam optimizer
        :param replay_type: the type of replay buffer, i.e., 'default' or 'prioritized'
        :param loss_type: the loss to use during gradient descent
        :param network_type: the network architecture to use for the value and target networks
        :param training: True if the agent is being trained, False otherwise
        """

        # Call the parent constructor.
        super().__init__(
            gamma=gamma, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size,
            learning_starts=learning_starts, target_update_interval=target_update_interval, adam_eps=adam_eps,
            replay_type=replay_type, loss_type=loss_type, network_type=network_type, training=training
        )
