from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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

    # Parse the script arguments.
    parser = ArgumentParser(prog="run_training", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--agent", type=str, default="DQN", help="name of the agent to train")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="name of the environment on which to train the agent")
    parser.add_argument("--seed", type=int, default=0, help="random seed to use")
    args = parser.parse_args()

    # Train a reinforcement learning agent on a gym environment.
    run_training(agent_name=args.agent, env_name=args.env, seed=args.seed)
