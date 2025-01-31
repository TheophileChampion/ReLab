from relab.agents.DQN import ReplayType, NetworkType, LossType, DQN


class MDQN(DQN):
    """!
    @brief Implements a multistep Deep Q-Network.

    @details
    This implementation is based on the paper:

    <b>Learning to predict by the methods of temporal differences</b>,
    published in Machine learning, 3:9–44, 1988.

    Authors:
    - Richard S. Sutton
    """

    def __init__(
        self,
        gamma: float = 0.99,
        learning_rate: float = 0.00001,
        buffer_size: int = 1000000,
        batch_size: int = 32,
        learning_starts: int = 200000,
        target_update_interval: int = 40000,
        adam_eps: float = 1.5e-4,
        n_actions: int = 18,
        n_atoms: int = 1,
        training: bool = True,
        n_steps: int = 3,
        replay_type: ReplayType = ReplayType.MULTISTEP,
        loss_type: LossType = LossType.DQN_SL1,
        network_type: NetworkType = NetworkType.DEFAULT,
    ) -> None:
        """!
        Create a DQN agent.
        @param gamma: the discount factor
        @param learning_rate: the learning rate
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param learning_starts: the step at which learning starts
        @param target_update_interval: number of training steps between two synchronization of the target
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param n_actions: the number of actions available to the agent
        @param n_atoms: the number of atoms used to approximate the distribution over returns
        @param training: True if the agent is being trained, False otherwise
        @param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        @param replay_type: the type of replay buffer
        @param loss_type: the loss to use during gradient descent
        @param network_type: the network architecture to use for the value and target networks
        """

        # Call the parent constructor.
        super().__init__(
            gamma=gamma,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            target_update_interval=target_update_interval,
            adam_eps=adam_eps,
            n_actions=n_actions,
            n_atoms=n_atoms,
            training=training,
            n_steps=n_steps,
            replay_type=replay_type,
            loss_type=loss_type,
            network_type=network_type,
        )
