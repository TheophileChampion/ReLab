from relab.agents.DQN import ReplayType, LossType, NetworkType, DQN


class RainbowDQN(DQN):
    """!
    Implement the rainbow Deep Q-Network agent from:
    Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot,
    Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning.
    In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.
    """

    def __init__(
        self, gamma=0.99, learning_rate=0.0000625, buffer_size=1000000, batch_size=32, learning_starts=80000, kappa=None,
        target_update_interval=32000, adam_eps=1.5e-4, n_actions=18, n_atoms=51, v_min=-10, v_max=10, training=True,
        n_steps=3, omega=0.5, replay_type=ReplayType.MULTISTEP_PRIORITIZED, loss_type=LossType.RAINBOW,
        network_type=NetworkType.RAINBOW
    ):
        """!
        Create a rainbow DQN agent.
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
        @param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        @param omega: the prioritization exponent
        @param replay_type: the type of replay buffer
        @param loss_type: the loss to use during gradient descent
        @param network_type: the network architecture to use for the value and target networks
        """

        # Call the parent constructor.
        super().__init__(
            gamma=gamma, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size,
            learning_starts=learning_starts, kappa=kappa, target_update_interval=target_update_interval,
            adam_eps=adam_eps, n_actions=n_actions, n_atoms=n_atoms, v_min=v_min, v_max=v_max, training=training,
            n_steps=n_steps, omega=omega, replay_type=replay_type, loss_type=loss_type, network_type=network_type,
            epsilon_schedule=[(0, 0)]
        )
