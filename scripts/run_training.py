from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from benchmarks import agents
import benchmarks
from benchmarks import environments


def run_training(agent, env, seed):
    """
    Train a reinforcement learning agent on a gym environment.
    :param agent: the agent name
    :param env: the environment name
    :param seed: the random seed
    """

    # Initialize the benchmark.
    benchmarks.initialize(agent, env, seed)

    # Create the environment.
    env = environments.make(env)

    # Create and train the agent.
    agent = agents.make(agent, training=True)
    agent.load()
    agent.train(env)


if __name__ == "__main__":

    # Parse the script arguments.
    parser = ArgumentParser(prog="run_training", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--agent", type=str, default="DDQN", help="name of the agent to train")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="name of the environment on which to train the agent")
    parser.add_argument("--seed", type=int, default=0, help="random seed to use")
    args = parser.parse_args()

    # Train a reinforcement learning agent on a gym environment.
    run_training(agent=args.agent, env=args.env, seed=args.seed)
