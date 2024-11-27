from benchmarks import agents
import benchmarks
from benchmarks import environments


def run_training(agent_name, env_name, seed):
    """
    Train a reinforcement learning agent on a gym environment.
    :param agent_name: the agent name
    :param env_name: the environment name
    :param seed: the random seed
    """

    # Initialize the benchmark.
    benchmarks.initialize(agent_name, env_name, seed)

    # Create the environment.
    env = environments.make(env_name)

    # Create and train the agent.
    agent = agents.make(agent_name, training=True)
    agent.load()
    agent.train(env)


if __name__ == "__main__":

    # Train a reinforcement learning agent on a gym environment.
    run_training(agent_name="DQN", env_name="ALE/Pong-v5", seed=0)
