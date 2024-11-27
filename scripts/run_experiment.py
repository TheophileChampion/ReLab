import os
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import re

import benchmarks
from benchmarks.environments import small_benchmark_atari_games as atari_games


def launch_job(task, kwargs, dependencies=None):
    """
    Launch a slurm job.
    :param task: the task to run, e.g., "run_training" or "run_demo"
    :param kwargs: the keywords argument of the job's shell script
    :param dependencies: the job indices on which this job depends
    :return: the index of the slurm job launched
    """

    # Create the job's command line.
    python_args = " ".join([f"--{key} {value}" for key, value in kwargs.items()])
    dependencies = "" if dependencies is None else f"-d afterok:{':'.join(dependencies)}"
    command = f"sbatch {dependencies} {task} {python_args}"

    # Launch the slurm job.
    process = subprocess.run(command.split(), capture_output=True, text=True)

    # Return the job index of the slurm job.
    return re.findall(r"\d+", process.stdout)[-1]


def run_experiment(agent_names, env_names, seeds):
    """
    Run an experiments by:
    - training all reinforcement learning agents on all gym environments using all seeds
    - for each triple (agent, env, seed), create a gif demonstrating the learned policy
    - for each environment, create a graph displaying the mean and standard deviation of the agent performance
    :param agent_names: the agent names
    :param env_names: the environment names
    :param seeds: the random seeds
    """

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
                job_id = launch_job(
                    task=prefix + os.sep + "run_training.sh",
                    kwargs={"agent": agent, "env": env, "seed": seed}
                )
                job_indices.append(job_id)

                # Demonstrate the policy learned by the agent on the environment with the specified seed.
                launch_job(
                    task=prefix + os.sep + "run_demo.sh",
                    kwargs={"agent": agent, "env": env, "seed": seed, "index": benchmarks.config(key="max_n_steps")},
                    dependencies=[job_id]
                )

        # Draw the graph of mean episodic reward for all agents in the current environment.
        launch_job(
            task=prefix + os.sep + "draw_graph.sh",
            kwargs={"agents": " ".join(agent_names), "env": env, "seeds": " ".join(seeds), "metric": "mean_episodic_reward"},
            dependencies=job_indices
        )
        job_indices.clear()


if __name__ == "__main__":

    # Parse the script arguments.
    parser = ArgumentParser(prog="run_experiments", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--agents", nargs="+", default=["DQN", "RainbowDQN", "RainbowIQN"], help="name of the agents to train")
    parser.add_argument("--envs", nargs="+", default=atari_games(), help="name of the environments on which to train the agents")
    parser.add_argument("--seeds", nargs="+", default=[str(i) for i in range(5)], help="random seeds to use")
    args = parser.parse_args()

    # Train a reinforcement learning agent on a gym environment.
    run_experiment(agent_names=args.agents, env_names=args.envs, seeds=args.seeds)
