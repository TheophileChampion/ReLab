#!/usr/bin/env python

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import relab
from relab import agents, environments


def run_demo(agent: str, env: str, seed: int, index: int):
    """
    Demonstrate the policy learned by a reinforcement learning agent on a gym environment.
    :param agent: the agent name
    :param env: the environment name
    :param seed: the random seed
    :param index: the number of training steps corresponding to the checkpoint to load
    """

    # Initialize the benchmark.
    relab.initialize(agent, env, seed)

    # Create the environment.
    env = environments.make(env, render_mode="rgb_array")

    # Create and train the agent.
    agent = agents.make(agent, training=False)
    agent.load(f"model_{index}.pt")
    agent.demo(env, f"demo_{index}.gif")


def main():
    """
    Entry point of the run_demo.py script.
    """

    # Parse the script arguments.
    parser = ArgumentParser(
        prog="run_demo.py", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="DuelingDDQN",
        help="name of the agent whose policy needs to be demonstrated",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ALE/Pong-v5",
        help="name of the environment on which to demonstrate the agent's policy",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed to use")
    parser.add_argument(
        "--index", type=int, default=10000000, help="index of the checkpoint to load"
    )
    args = parser.parse_args()

    # Demonstrate the policy learned by a reinforcement learning agent on a gym environment.
    run_demo(agent=args.agent, env=args.env, seed=args.seed, index=args.index)


if __name__ == "__main__":
    main()
