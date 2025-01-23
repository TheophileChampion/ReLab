from relab.agents.DQN import DQN, LossType, ReplayType, NetworkType


class DuelingDQN(DQN):
    """!
    Implement the Dueling Deep Q-Network agent from:
    Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando Freitas.
    Dueling network architectures for deep reinforcement learning.
    In International conference on machine learning. PMLR, 2016.
    """

    def __init__(
        self, gamma=0.99, learning_rate=0.00001, buffer_size=1000000, batch_size=32, learning_starts=200000,
        target_update_interval=40000, adam_eps=1.5e-4, replay_type=ReplayType.DEFAULT, loss_type=LossType.DQN_SL1,
        network_type=NetworkType.DUELING, training=True
    ):
        """!
        Create a Dueling DQN agent.
        @param gamma: the discount factor
        @param learning_rate: the learning rate
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param learning_starts: the step at which learning starts
        @param target_update_interval: number of training steps between two synchronization of the target
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param replay_type: the type of replay buffer
        @param loss_type: the loss to use during gradient descent
        @param network_type: the network architecture to use for the value and target networks
        @param training: True if the agent is being trained, False otherwise
        """

        # Call the parent constructor.
        super().__init__(
            gamma=gamma, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size,
            learning_starts=learning_starts, target_update_interval=target_update_interval, adam_eps=adam_eps,
            replay_type=replay_type, loss_type=loss_type, network_type=network_type, training=training
        )
