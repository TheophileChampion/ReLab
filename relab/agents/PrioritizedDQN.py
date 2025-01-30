from relab.agents.DQN import DQN, LossType, ReplayType, NetworkType


class PrioritizedDQN(DQN):
    """!
    @brief Implement a DQN with prioritized replay buffer.

    @details
    The implementation is based on the following papers:

    <b>Prioritized experience replay</b>,
    published on arXiv, 2015.

    Authors:
    - Tom Schaul

    More precisely, prioritized DQN improves upon DQN by tracking the loss associated with
    each experience in the replay buffer. During training, experiences are sampled in proportion
    to their losses, ensuring that experiences with higher losses are replayed more frequently.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        learning_rate: float = 0.00025,
        buffer_size: int = 1000000,
        batch_size: int = 32,
        learning_starts: int = 200000,
        target_update_interval: int = 40000,
        adam_eps: float = 1.5e-4,
        replay_type: ReplayType = ReplayType.PRIORITIZED,
        loss_type: LossType = LossType.DQN_SL1,
        network_type: NetworkType = NetworkType.DEFAULT,
        omega: float = 0.7,
        omega_is: float = 0.5,
        training: bool = True
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
        @param replay_type: the type of replay buffer, i.e., 'default' or 'prioritized'
        @param loss_type: the loss to use during gradient descent
        @param network_type: the network architecture to use for the value and target networks
        @param omega: the prioritization exponent
        @param omega_is: the important sampling exponent
        @param training: True if the agent is being trained, False otherwise
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
            replay_type=replay_type,
            loss_type=loss_type,
            network_type=network_type,
            training=training,
            omega=omega,
            omega_is=omega_is
        )
