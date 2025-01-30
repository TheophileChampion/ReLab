from typing import Any

from relab.agents.OldVAE import VAE as OldVAE  # TODO
from relab.agents.AgentInterface import AgentInterface
from relab.agents.BetaHMM import BetaHMM
from relab.agents.BetaVAE import BetaVAE
from relab.agents.CDQN import CDQN
from relab.agents.DiscreteHMM import DiscreteHMM
from relab.agents.DiscreteVAE import DiscreteVAE
from relab.agents.DuelingDQN import DuelingDQN
from relab.agents.DDQN import DDQN
from relab.agents.DQN import DQN
from relab.agents.DuelingDDQN import DuelingDDQN
from relab.agents.HMM import HMM
from relab.agents.IQN import IQN
from relab.agents.JointHMM import JointHMM
from relab.agents.JointVAE import JointVAE
from relab.agents.MDQN import MDQN
from relab.agents.NoisyCDQN import NoisyCDQN
from relab.agents.NoisyDDQN import NoisyDDQN
from relab.agents.NoisyDQN import NoisyDQN
from relab.agents.PrioritizedDDQN import PrioritizedDDQN
from relab.agents.PrioritizedDQN import PrioritizedDQN
from relab.agents.PrioritizedMDQN import PrioritizedMDQN
from relab.agents.QRDQN import QRDQN
from relab.agents.RainbowDQN import RainbowDQN
from relab.agents.RainbowIQN import RainbowIQN
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
        "Random": Random,
        "QRDQN": QRDQN,
        "DDQN": DDQN,
        "CDQN": CDQN,
        "MDQN": MDQN,
        "IQN": IQN,
        "DQN": DQN,
        "DiscreteVAE": DiscreteVAE,
        "JointVAE": JointVAE,
        "BetaVAE": BetaVAE,
        "VAE": VAE,
        "DiscreteHMM": DiscreteHMM,
        "JointHMM": JointHMM,
        "BetaHMM": BetaHMM,
        "HMM": HMM,
        "OldVAE": OldVAE,  # TODO
    }

    # Check if the agent is supported, raise an error if it isn't.
    if agent_name not in agents.keys():
        raise RuntimeError(f"[Error]: agent {agent_name} not supported.")

    # Create an instance of the requested agent.
    return agents[agent_name](**kwargs)
