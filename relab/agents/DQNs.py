from relab.agents.DQN import DQN, LossType, NetworkType, ReplayType


class CDQN(DQN):
    """!
    @brief Implements a Categorical Deep Q-Network.

    @details
    This implementation is based on the paper:

    <b>A Distributional Perspective on Reinforcement Learning</b>,
    published in PMLR, 2017.

    Authors:
    - Marc G. Bellemare
    - Will Dabney
    - Rémi Munos

    The paper introduced the CDQN, which takes a distributional perspective on value-based reinforcement
    learning, by learning a categorical distribution over returns instead of the expected returns.
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
        n_atoms: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        training: bool = True,
        replay_type: ReplayType = ReplayType.DEFAULT,
        loss_type: LossType = LossType.KL_DIVERGENCE,
        network_type: NetworkType = NetworkType.CATEGORICAL,
    ) -> None:
        """!
        Create a categorical DQN agent.
        @param gamma: the discount factor
        @param learning_rate: the learning rate
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param learning_starts: the step at which learning starts
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
            gamma=gamma,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            target_update_interval=target_update_interval,
            adam_eps=adam_eps,
            n_actions=n_actions,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            training=training,
            replay_type=replay_type,
            loss_type=loss_type,
            network_type=network_type,
        )


class DDQN(DQN):
    """!
    @brief Implements a Double Deep Q-Network.

    @details
    This implementation is based on the paper:

    <b>Deep Reinforcement Learning with Double Q-learning</b>,
    published in AAAI, 2016.

    Authors:
    - Hado van Hasselt
    - Arthur Guez
    - David Silver

    The paper introduced Double Q-learning to Deep Q-Networks to reduce
    overestimation bias by decoupling action selection from action evaluation.
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
        replay_type: ReplayType = ReplayType.DEFAULT,
        loss_type: LossType = LossType.DDQN_SL1,
        network_type: NetworkType = NetworkType.DEFAULT,
        training: bool = True,
    ) -> None:
        """!
        Create a Double DQN agent.
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
        )


class DuelingDDQN(DQN):
    """!
    @brief Implements a Dueling Double Deep Q-Network.

    @details
    For more information about the original papers, please refer to the documentation of DDQN and DuelingDQN.
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
        replay_type: ReplayType = ReplayType.DEFAULT,
        loss_type: LossType = LossType.DDQN_SL1,
        network_type: NetworkType = NetworkType.DUELING,
        training: bool = True,
    ) -> None:
        """!
        Create a Dueling Double DQN agent.
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
        )


class DuelingDQN(DQN):
    """!
    @brief Implements a Dueling Deep Q-Network.

    @details
    This implementation is based on the paper:

    <b>Dueling network architectures for deep reinforcement learning</b>,
    published in PMLR, 2016.

    Authors:
    - Ziyu Wang
    - Tom Schaul
    - Matteo Hessel
    - Hado Hasselt
    - Marc Lanctot
    - Nando Freitas

    More precisely, the DuelingDQN architecture improves the standard DQN by separating the
    representation of state values and action advantages.
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
        replay_type: ReplayType = ReplayType.DEFAULT,
        loss_type: LossType = LossType.DQN_SL1,
        network_type: NetworkType = NetworkType.DUELING,
        training: bool = True,
    ) -> None:
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
        )


class IQN(DQN):
    """!
    @brief Implement an Implicit Quantile Network.

    @details
    This implementation is based on the paper:

    <b>Implicit quantile networks for distributional reinforcement learning</b>,
    published in PMLR, 2018.

    Authors:
    - Will Dabney
    - Georg Ostrovski
    - David Silver
    - Rémi Munos

    The paper introduced the Implicit Quantile Network (IQN), which combines quantile regression with
    distributional deep reinforcement learning. Importantly, the number of quantiles is independent
    of the network size, and can be adjusted depending on the amount of compute available.
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
        replay_type: ReplayType = ReplayType.DEFAULT,
        loss_type: LossType = LossType.IMPLICIT_QUANTILE,
        network_type: NetworkType = NetworkType.IMPLICIT_QUANTILE,
    ) -> None:
        """!
        Create an IQN agent.
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
            replay_type=replay_type,
            loss_type=loss_type,
            network_type=network_type,
        )


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


class NoisyCDQN(DQN):
    """!
    @brief Implement a Categorical Deep Q-Network with noisy linear layers (NoisyCDQN).

    @details
    For more information about the original papers, please refer to the documentation of CDQN and NoisyDQN.
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
        n_atoms: int = 21,
        v_min: float = -10,
        v_max: float = 10,
        training: bool = True,
        replay_type: ReplayType = ReplayType.DEFAULT,
        loss_type: LossType = LossType.KL_DIVERGENCE,
        network_type: NetworkType = NetworkType.NOISY_CATEGORICAL,
    ) -> None:
        """!
        Create a categorical DQN agent.
        @param gamma: the discount factor
        @param learning_rate: the learning rate
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param learning_starts: the step at which learning starts
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
            gamma=gamma,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            target_update_interval=target_update_interval,
            adam_eps=adam_eps,
            n_actions=n_actions,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            training=training,
            replay_type=replay_type,
            loss_type=loss_type,
            network_type=network_type,
        )


class NoisyDDQN(DQN):
    """!
    @brief Implement a Double DQN with noisy linear layers.

    @details
    For more information about the original papers, please refer to the documentation of DDQN and NoisyDQN.
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
        replay_type: ReplayType = ReplayType.DEFAULT,
        loss_type: LossType = LossType.DDQN_SL1,
        network_type: NetworkType = NetworkType.NOISY,
        training: bool = True,
    ) -> None:
        """!
        Create a Double DQN agent.
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
        )


class NoisyDQN(DQN):
    """!
    @brief Implement a DQN with noisy linear layers.

    @details
    For more information about the original papers, please refer to the documentation of DQN and NoisyDQN.
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
        replay_type: ReplayType = ReplayType.DEFAULT,
        loss_type: LossType = LossType.DQN_SL1,
        network_type: NetworkType = NetworkType.NOISY,
        training: bool = True,
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
        @param replay_type: the type of replay buffer
        @param loss_type: the loss to use during gradient descent
        @param network_type: the network architecture to use for the value and target networks
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
        )


class PrioritizedDDQN(DQN):
    """!
    @brief Implement a Double DQN with prioritized replay buffer.

    @details
    For more information about the original papers, please refer to the documentation of DDQN and PrioritizedDQN.
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
        loss_type: LossType = LossType.DDQN_SL1,
        network_type: NetworkType = NetworkType.DEFAULT,
        omega: float = 0.7,
        omega_is: float = 0.5,
        training: bool = True,
    ) -> None:
        """!
        Create a Double DQN agent.
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
            omega_is=omega_is,
        )


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
        training: bool = True,
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
            omega_is=omega_is,
        )


class PrioritizedMDQN(DQN):
    """!
    @brief Implement a multistep DQN with prioritized replay buffer.

    @details
    For more information about the original papers, please refer to the documentation of MDQN and PrioritizedDQN.
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
        omega: float = 0.7,
        omega_is: float = 0.5,
        replay_type: ReplayType = ReplayType.MULTISTEP_PRIORITIZED,
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
        @param omega: the prioritization exponent
        @param omega_is: the important sampling exponent
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
            omega=omega,
            omega_is=omega_is,
        )


class QRDQN(DQN):
    """!
    @brief Implement a quantile regression Deep Q-Network.

    @details
    This implementation is based on the paper:

    <b>Distributional reinforcement learning with quantile regression</b>,
    published in AAAI, 2018.

    Authors:
    - Will Dabney
    - Mark Rowland
    - Marc Bellemare
    - Rémi Munos

    The paper introduced the quantile regression DQN, which combines quantile regression with distributional
    deep reinforcement learning. Importantly, the number of quantiles dependent on the network's output size.
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
        n_atoms: int = 32,
        training: bool = True,
        replay_type: ReplayType = ReplayType.DEFAULT,
        loss_type: LossType = LossType.QUANTILE,
        network_type: NetworkType = NetworkType.QUANTILE,
    ) -> None:
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
        @param training: True if the agent is being trained, False otherwise
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
            replay_type=replay_type,
            loss_type=loss_type,
            network_type=network_type,
        )


class RainbowDQN(DQN):
    """!
    @brief Implement a rainbow Deep Q-Network.

    @details
    This implementation is based on the paper:

    <b>Rainbow: Combining improvements in deep reinforcement learning</b>,
    published in AAAI, 2018.

    Authors:
    - Matteo Hessel
    - Joseph Modayil
    - Hado Van Hasselt
    - Tom Schaul
    - Georg Ostrovski
    - Will Dabney
    - Dan Horgan
    - Bilal Piot
    - Mohammad Azar
    - David Silver

    The paper introduced RainbowDQN which combines the following improvements of DQN:
    - double Q-learning
    - multistep Q-learning
    - distributional reinforcement learning
    - noisy layers for exploration
    - dueling DQN
    - prioritized replay buffer
    """

    def __init__(
        self,
        gamma: float = 0.99,
        learning_rate: float = 0.0000625,
        buffer_size: int = 1000000,
        batch_size: int = 32,
        learning_starts: int = 80000,
        target_update_interval: int = 32000,
        adam_eps: float = 1.5e-4,
        n_actions: int = 18,
        n_atoms: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        training: bool = True,
        n_steps: int = 3,
        omega: float = 0.5,
        replay_type: ReplayType = ReplayType.MULTISTEP_PRIORITIZED,
        loss_type: LossType = LossType.RAINBOW,
        network_type: NetworkType = NetworkType.RAINBOW,
    ) -> None:
        """!
        Create a rainbow DQN agent.
        @param gamma: the discount factor
        @param learning_rate: the learning rate
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param learning_starts: the step at which learning starts
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
            gamma=gamma,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            target_update_interval=target_update_interval,
            adam_eps=adam_eps,
            n_actions=n_actions,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            training=training,
            n_steps=n_steps,
            omega=omega,
            replay_type=replay_type,
            loss_type=loss_type,
            network_type=network_type,
            epsilon_schedule=[(0, 0)],
        )


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
        n_steps: int = 3,
        omega: float = 0.5,
        replay_type: ReplayType = ReplayType.MULTISTEP_PRIORITIZED,
        loss_type: LossType = LossType.RAINBOW_IQN,
        network_type: NetworkType = NetworkType.RAINBOW_IQN,
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
            epsilon_schedule=[(0, 0)],
        )
