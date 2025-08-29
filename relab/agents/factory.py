from typing import Any

from relab.agents.A2C import A2C
from relab.agents.AgentInterface import AgentInterface
from relab.agents.BetaHMM import BetaHMM
from relab.agents.BetaVAE import BetaVAE
from relab.agents.CHMM import CHMM
from relab.agents.DQN import DQN
from relab.agents.DQNs import (
    CDQN,
    DDQN,
    IQN,
    MDQN,
    QRDQN,
    DuelingDDQN,
    DuelingDQN,
    NoisyCDQN,
    NoisyDDQN,
    NoisyDQN,
    PrioritizedDDQN,
    PrioritizedDQN,
    PrioritizedMDQN,
    RainbowDQN,
    RainbowIQN,
)
from relab.agents.HMM import HMM
from relab.agents.Human import Human
from relab.agents.PG import PG
from relab.agents.PPO import PPO
from relab.agents.Random import Random
from relab.agents.VAE import VAE


def make(agent_name: str, **kwargs: Any) -> AgentInterface:
    """!
    Create the agent whose name is passed as parameters.
    @param agent_name: the name of the agent to instantiate
    @param kwargs: keyword arguments to pass to the agent constructor
    @return the created agent
    """

    # The lists of all supported agents.
    agents = {
        "PrioritizedMDQN": PrioritizedMDQN,
        "PrioritizedDDQN": PrioritizedDDQN,
        "PrioritizedDQN": PrioritizedDQN,
        "DuelingDDQN": DuelingDDQN,
        "DuelingDQN": DuelingDQN,
        "RainbowDQN": RainbowDQN,
        "RainbowIQN": RainbowIQN,
        "NoisyCDQN": NoisyCDQN,
        "NoisyDDQN": NoisyDDQN,
        "NoisyDQN": NoisyDQN,
        "BetaVAE": BetaVAE,
        "BetaHMM": BetaHMM,
        "Random": Random,
        "Human": Human,
        "QRDQN": QRDQN,
        "DDQN": DDQN,
        "CDQN": CDQN,
        "MDQN": MDQN,
        "IQN": IQN,
        "DQN": DQN,
        "CHMM": CHMM,
        "VAE": VAE,
        "HMM": HMM,
        "PPO": PPO,
        "A2C": A2C,
        "PG": PG,
    }

    # Check if the agent is supported, raise an error if it isn't.
    if agent_name not in agents.keys():
        raise RuntimeError(f"[Error]: agent {agent_name} not supported.")

    # Create an instance of the requested agent.
    return agents[agent_name](**kwargs)
