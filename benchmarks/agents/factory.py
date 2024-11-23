from benchmarks.agents.CategoricalDQN import CategoricalDQN
from benchmarks.agents.DuelingDQN import DuelingDQN
from benchmarks.agents.DDQN import DDQN
from benchmarks.agents.DQN import DQN
from benchmarks.agents.DuelingDDQN import DuelingDDQN
from benchmarks.agents.NoisyCategoricalDQN import NoisyCategoricalDQN
from benchmarks.agents.NoisyDDQN import NoisyDDQN
from benchmarks.agents.NoisyDQN import NoisyDQN
from benchmarks.agents.PrioritizedDDQN import PrioritizedDDQN
from benchmarks.agents.PrioritizedDQN import PrioritizedDQN


def make(agent_name, **kwargs):
    """
    Create the agent whose name is required as parameters.
    :param agent_name: the name of the agent to instantiate
    :param kwargs: keyword arguments to pass to the agent constructor
    :return: the created agent
    """

    # The lists of all supported agents.
    agents = {
        "DQN": DQN,
        "NoisyDQN": NoisyDQN,
        "DuelingDQN": DuelingDQN,
        "CDQN": CategoricalDQN,
        "PrioritizedDQN": PrioritizedDQN,
        "DDQN": DDQN,
        "NoisyDDQN": NoisyDDQN,
        "DuelingDDQN": DuelingDDQN,
        "NoisyCDQN": NoisyCategoricalDQN,
        "PrioritizedDDQN": PrioritizedDDQN,
    }

    # Check if the agent is supported, raise an error if it isn't.
    if agent_name not in agents.keys():
        raise RuntimeError(f"[Error]: agent {agent_name} not supported.")

    # Create an instance of the requested agent.
    return agents[agent_name](**kwargs)
