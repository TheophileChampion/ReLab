#!/usr/bin/env python

import argparse
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from copy import deepcopy
from functools import partial
from typing import List

import relab
from relab.environments import small_atari_benchmark as atari_games
from relab.helpers.JobRunners import LocalJobRunner, SlurmJobRunner


def run_experiment(
    agent_names: List[str], env_names: List[str], seeds: List[int], local: bool = True
) -> None:
    """
    Run an experiments by:
    - training all reinforcement learning agents on all gym environments using all seeds
    - for each triple (agent, env, seed), create a gif demonstrating the learned policy
    - for each environment, create a graph displaying the mean and standard deviation of the agent performance
    :param agent_names: the agent names
    :param env_names: the environment names
    :param seeds: the random seeds
    :param local: True for launching jobs on the local computer, False to launch slurm jobs
    """

    # Select the requested job runner.
    job_runners = {True: partial(LocalJobRunner, max_worker=5), False: SlurmJobRunner}
    job_runner = job_runners[local]()

    # Iterate over all environments.
    prefix = "." + os.sep + "scripts" + os.sep + "slurm"
    for env in env_names:

        # Keep track of all the training job indices on the current environment.
        job_indices = []

        # Iterate over all agents.
        for agent in agent_names:

            # Iterate over all seeds.
            for seed in seeds:

                # Train the agent on the environment with the specified seed.
                job_id = job_runner.launch_job(
                    task=prefix + os.sep + "run_training.sh",
                    kwargs={"agent": agent, "env": env, "seed": seed},
                )
                job_indices.append(job_id)

                # Demonstrate the policy learned by the agent on the environment with the specified seed.
                job_runner.launch_job(
                    task=prefix + os.sep + "run_demo.sh",
                    kwargs={
                        "agent": agent,
                        "env": env,
                        "seed": seed,
                        "index": relab.config(key="max_n_steps"),
                    },
                    dependencies=[job_id],
                )

        # Draw the graph of mean episodic reward for all agents in the current environment.
        job_runner.launch_job(
            task=prefix + os.sep + "draw_graph.sh",
            kwargs={
                "agents": agent_names,
                "env": env,
                "seeds": seeds,
                "metric": "mean_episodic_reward",
            },
            dependencies=deepcopy(job_indices),
        )

    # Wait for all job to terminate.
    job_runner.wait()


def main():
    """
    Entry point of the run_experiments script.
    """

    # Parse the script arguments.
    parser = ArgumentParser(
        prog="run_experiments", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["DQN", "RainbowDQN", "RainbowIQN"],
        help="name of the agents to train",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=atari_games(),
        help="name of the environments on which to train the agents",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[str(i) for i in range(5)],
        help="random seeds to use",
    )
    parser.add_argument("--local", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    # Train a reinforcement learning agent on a gym environment.
    run_experiment(
        agent_names=args.agents, env_names=args.envs, seeds=args.seeds, local=args.local
    )


if __name__ == "__main__":
    main()
