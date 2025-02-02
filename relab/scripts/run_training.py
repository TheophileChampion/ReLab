#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from relab import agents
import relab
from relab import environments


def run_training(agent : str, env : str, seed : int) -> None:
    """
    Train a reinforcement learning agent on a gym environment.
    :param agent: the agent name
    :param env: the environment name
    :param seed: the random seed
    """

    # Initialize the benchmark.
    relab.initialize(agent, env, seed)

    # Create the environment.
    env = environments.make(env)

    # Create and train the agent.
    agent = agents.make(agent, training=True)
    agent.load()
    agent.train(env)


def main():
    """
    Entry point of the run_training.py script.
    """

    # Parse the script arguments.
    parser = ArgumentParser(prog="run_training.py", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--agent", type=str, default="DDQN", help="name of the agent to train")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="name of the environment on which to train the agent")
    parser.add_argument("--seed", type=int, default=0, help="random seed to use")
    args = parser.parse_args()

    # Train a reinforcement learning agent on a gym environment.
    run_training(agent=args.agent, env=args.env, seed=args.seed)


if __name__ == "__main__":
    main()
