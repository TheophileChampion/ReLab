from relab.agents.DQN import DQN, LossType, ReplayType, NetworkType


class RainbowIQN(DQN):
    """!
    @brief Implement a rainbow implicit quantile network.

    @details
    This implementation is based on the paper:

    <b>Is deep reinforcement learning really superhuman on atari? leveling the playing field</b>,
    published in arXiv, 2019.

    Authors:
    - Marin Toromanoff
    - Emilie Wirbel
    - Fabien Moutarde

    The paper introduced RainbowIQN which improves upon RainbowDQN by replacing
    its distributional reinforcement learning part by an implicit quantile network.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        learning_rate: float = 0.00001,
        buffer_size: int = 1000000,
        batch_size: int = 32,
        learning_starts: int = 200000,
        kappa: float = 1.0,
        target_update_interval: int = 40000,
        adam_eps: float = 1.5e-4,
        n_actions: int = 18,
        n_atoms: int = 8,
        training: bool = True,
        n_steps: int = 3, omega: float = 0.5,
        replay_type: ReplayType = ReplayType.MULTISTEP_PRIORITIZED,
        loss_type: LossType = LossType.RAINBOW_IQN,
        network_type: NetworkType = NetworkType.RAINBOW_IQN
    ) -> None:
        """!
        Create a rainbow IQN agent.
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
        @param training: True if the agent is being trained, False otherwise
        @param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        @param omega: the prioritization exponent
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
            kappa=kappa,
            learning_starts=learning_starts,
            target_update_interval=target_update_interval,
            adam_eps=adam_eps,
            n_actions=n_actions,
            n_atoms=n_atoms,
            training=training,
            n_steps=n_steps,
            omega=omega,
            replay_type=replay_type,
            loss_type=loss_type,
            network_type=network_type,
            epsilon_schedule=[(0, 0)]
        )
