from benchmarks.agents.BetaHMM import BetaHMM
from benchmarks.agents.BetaVAE import BetaVAE
from benchmarks.agents.CDQN import CDQN
from benchmarks.agents.DiscreteHMM import DiscreteHMM
from benchmarks.agents.DiscreteVAE import DiscreteVAE
from benchmarks.agents.DuelingDQN import DuelingDQN
from benchmarks.agents.DDQN import DDQN
from benchmarks.agents.DQN import DQN
from benchmarks.agents.DuelingDDQN import DuelingDDQN
from benchmarks.agents.HMM import HMM
from benchmarks.agents.IQN import IQN
from benchmarks.agents.JointHMM import JointHMM
from benchmarks.agents.JointVAE import JointVAE
from benchmarks.agents.MDQN import MDQN
from benchmarks.agents.NoisyCDQN import NoisyCDQN
from benchmarks.agents.NoisyDDQN import NoisyDDQN
from benchmarks.agents.NoisyDQN import NoisyDQN
from benchmarks.agents.PrioritizedDDQN import PrioritizedDDQN
from benchmarks.agents.PrioritizedDQN import PrioritizedDQN
from benchmarks.agents.PrioritizedMDQN import PrioritizedMDQN
from benchmarks.agents.QRDQN import QRDQN
from benchmarks.agents.RainbowDQN import RainbowDQN
from benchmarks.agents.RainbowIQN import RainbowIQN
from benchmarks.agents.Random import Random
from benchmarks.agents.VAE import VAE


def make(agent_name, **kwargs):
    """
    Create the agent whose name is passed as parameters.
    :param agent_name: the name of the agent to instantiate
    :param kwargs: keyword arguments to pass to the agent constructor
    :return: the created agent
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
    }

    # Check if the agent is supported, raise an error if it isn't.
    if agent_name not in agents.keys():
        raise RuntimeError(f"[Error]: agent {agent_name} not supported.")

    # Create an instance of the requested agent.
    return agents[agent_name](**kwargs)
