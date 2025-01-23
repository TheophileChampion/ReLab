from relab.agents.AgentInterface import ReplayType
from relab.agents.HMM import HMM
from relab.agents.VariationalModel import LikelihoodType, LatentSpaceType


class DiscreteHMM(HMM):
    """!
    Implement an agent taking random actions, and learning a world model using a HMM with discrete latent space.
    """

    def __init__(
        self, learning_starts=200000, n_actions=18, training=True,
        likelihood_type=LikelihoodType.BERNOULLI, latent_space_type=LatentSpaceType.DISCRETE,
        n_continuous_vars=10, n_discrete_vars=20, n_discrete_vals=10,
        learning_rate=0.00001, adam_eps=1.5e-4, beta_schedule=None, tau_schedule=None,
        replay_type=ReplayType.DEFAULT, buffer_size=1000000, batch_size=32, n_steps=1, omega=1.0, omega_is=1.0
    ):
        """!
        Create a Hidden Markov Model agent taking random actions.
        @param learning_starts: the step at which learning starts
        @param n_actions: the number of actions available to the agent
        @param training: True if the agent is being trained, False otherwise
        @param likelihood_type: the type of likelihood used by the world model
        @param latent_space_type: the type of latent space used by the world model
        @param n_continuous_vars: the number of continuous latent variables
        @param n_discrete_vars: the number of discrete latent variables
        @param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        @param learning_rate: the learning rate
        @param adam_eps: the epsilon parameter of the Adam optimizer
        @param beta_schedule: the piecewise linear schedule of the KL-divergence weight of beta-VAE
        @param tau_schedule: the exponential schedule of the temperature of the Gumbel-softmax
        @param replay_type: the type of replay buffer
        @param buffer_size: the size of the replay buffer
        @param batch_size: the size of the batches sampled from the replay buffer
        @param n_steps: the number of steps for which rewards are accumulated in multistep Q-learning
        @param omega: the prioritization exponent
        @param omega_is: the important sampling exponent
        """

        # Call the parent constructor.
        super().__init__(
            learning_starts=learning_starts, n_actions=n_actions, training=training,  replay_type=replay_type,
            likelihood_type=likelihood_type, latent_space_type=latent_space_type,
            buffer_size=buffer_size, batch_size=batch_size, n_steps=n_steps, omega=omega, omega_is=omega_is,
            n_continuous_vars=n_continuous_vars, n_discrete_vars=n_discrete_vars, n_discrete_vals=n_discrete_vals,
            learning_rate=learning_rate, adam_eps=adam_eps, beta_schedule=beta_schedule, tau_schedule=tau_schedule,
        )
