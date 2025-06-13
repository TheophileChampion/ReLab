from typing import Any

from relab.agents.AgentInterface import ReplayType
from relab.agents.HMM import HMM
from relab.agents.VariationalModel import LikelihoodType


class BetaHMM(HMM):
    """!
    @brief Implements a Beta Hidden Markov Model.

    @details
    This implementation extends upon the paper:

    <b>beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework</b>,
    published in ICLR, 2017.

    Authors:
    - Irina Higgins
    - Loic Matthey
    - Arka Pal
    - Christopher Burgess
    - Xavier Glorot
    - Matthew Botvinick
    - Shakir Mohamed
    - Alexander Lerchner

    More precisely, the β-HMM extends the β-VAE framework to sequential data
    by modeling temporal dependencies using a transition network.
    Note, this agent takes random actions.
    """

    def __init__(
        self,
        learning_starts: int = 50000,
        n_actions: int = 18,
        training: bool = True,
        likelihood_type: LikelihoodType = LikelihoodType.BERNOULLI,
        n_continuous_vars: int = 15,
        learning_rate: float = 0.0001,
        adam_eps: float = 1e-8,
        beta_schedule: Any = None,
        replay_type: ReplayType = ReplayType.PRIORITIZED,
        buffer_size: int = 1000000,
        batch_size: int = 50,
        n_steps: int = 1,
        omega: float = 1.0,
        omega_is: float = 1.0,
    ) -> None:
        """!
        Create a Hidden Markov Model agent taking random actions.
        @param learning_starts: the step at which learning starts
        @param n_actions: the number of actions available to the agent
        @param training: True if the agent is being trained, False otherwise
        @param likelihood_type: the type of likelihood used by the world model
        @param n_continuous_vars: the number of continuous latent variables
        @param learning_rate: the learning rate
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param beta_schedule: the piecewise linear schedule of the KL-divergence weight of beta-VAE
        @param replay_type: the type of replay buffer
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        @param omega: the prioritization exponent
        @param omega_is: the important sampling exponent
        """

        # Create the beta schedule.
        if beta_schedule is None:
            beta_schedule = [(0, 0.0001)]

        # Call the parent constructor.
        super().__init__(
            learning_starts=learning_starts,
            n_actions=n_actions,
            training=training,
            replay_type=replay_type,
            likelihood_type=likelihood_type,
            buffer_size=buffer_size,
            batch_size=batch_size,
            n_steps=n_steps,
            omega=omega,
            omega_is=omega_is,
            n_continuous_vars=n_continuous_vars,
            learning_rate=learning_rate,
            adam_eps=adam_eps,
            beta_schedule=beta_schedule,
        )
