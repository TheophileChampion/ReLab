from relab.agents.DQN import DQN, LossType, ReplayType, NetworkType


class QRDQN(DQN):
    """!
    Implement the quantile regression Deep Q-Network with the Huber quantile loss agent from:
    Will Dabney, Mark Rowland, Marc Bellemare, and Rémi Munos.
    Distributional reinforcement learning with quantile regression.
    In Proceedings of the AAAI conference on artificial intelligence, 2018.
    """

    def __init__(
        self, gamma=0.99, learning_rate=0.00001, buffer_size=1000000, batch_size=32, learning_starts=200000, kappa=1.0,
        target_update_interval=40000, adam_eps=1.5e-4, n_actions=18, n_atoms=32, v_min=None, v_max=None, training=True,
        replay_type=ReplayType.DEFAULT, loss_type=LossType.QUANTILE, network_type=NetworkType.QUANTILE
    ):
        """!
        Create a quantile regression DQN agent.
        @param gamma: the discount factor
        @param learning_rate: the learning rate
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param learning_starts: the step at which learning starts
        @param kappa: the kappa parameter of the quantile Huber loss see Equation (10) in QR-DQN paper
        @param target_update_interval: number of training steps between two synchronization of the target
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param n_actions: the number of actions available to the agent
        @param n_atoms: the number of atoms used to approximate the distribution over returns
        @param v_min: the minimum amount of returns (only used for categorical DQN)
        @param v_max: the maximum amount of returns (only used for categorical DQN)
        @param training: True if the agent is being trained, False otherwise
        @param replay_type: the type of replay buffer
        @param loss_type: the loss to use during gradient descent
        @param network_type: the network architecture to use for the value and target networks
        """

        # Call the parent constructor.
        super().__init__(
            gamma=gamma, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size, kappa=kappa,
            learning_starts=learning_starts, target_update_interval=target_update_interval, adam_eps=adam_eps,
            n_actions=n_actions, n_atoms=n_atoms, v_min=v_min, v_max=v_max, training=training,
            replay_type=replay_type, loss_type=loss_type, network_type=network_type
        )
