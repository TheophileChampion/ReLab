#!/usr/bin/env python

import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from os.path import exists, join
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import relab
import seaborn as sns
from pandas import DataFrame
from relab.helpers.FileSystem import FileSystem
from relab.helpers.MatPlotLib import MatPlotLib
from relab.helpers.TensorBoard import TensorBoard


def display_name(metric: str) -> str:
    """
    Retrieves the display name corresponding to the metric.
    :param metric: the metric name
    :return: the display name
    """
    return {
        "mean_time_elapsed_ms": "Time Per Training Iteration (ms)",
        "mean_memory_gb": "Memory Usage (GB)",
        "mean_episodic_reward": "Reward",
        "mean_episode_length": "Episode Length",
    }[metric]


def compute_summary_statistics(
    agent_name: str,
    env_name: str,
    seeds: List[int],
    metric: str,
    summary_statistics_path: str,
) -> DataFrame:
    """
    Compute the mean and standard deviations of the metric over all seeds.
    :param agent_name: the agent for which the mean and standard deviation of the metric is computed
    :param env_name: the environment for which the mean and standard deviation of the metric is computed
    :param seeds: the seeds
    :param metric: the metric name
    :param summary_statistics_path: the path where the summary statistics should be stored
    :return: a dataframe (with columns "step", "mean", and "std")
    """

    # Extract the metric values for each seed.
    summary_statistics = []
    for seed in seeds:

        # Extract the metric values of the current seed.
        relab.initialize(agent_name, env_name, seed, paths_only=True)
        metric_values = TensorBoard.load_log_directory(
            os.environ["TENSORBOARD_DIRECTORY"], metric
        )
        if metric_values is None:
            continue
        summary_statistics.append(metric_values)

    # Group them by training iteration.
    summary_statistics = pd.concat(summary_statistics, ignore_index=True, sort=False)
    summary_statistics = summary_statistics.groupby("step", as_index=False)

    # Compute the mean and standard deviation of the metric values.
    log_interval = relab.config("tensorboard_log_interval")
    mean = summary_statistics.mean().rename(columns={metric: "mean"})
    mean = mean[mean.step % log_interval == 0]
    std = summary_statistics.std().rename(columns={metric: "std"})
    std = std[std.step % log_interval == 0]
    if len(seeds) == 1:
        std = std.fillna(0)
    summary_statistics = mean.merge(std, on=["step", "step"]).dropna()

    # Save the summary statistics on the file system before.
    FileSystem.create_directory_and_file(summary_statistics_path)
    summary_statistics.to_csv(summary_statistics_path, sep="\t", index=False)
    return summary_statistics


def draw_graph(
    agents: List[str], env: str, seeds: List[int], metric: str, overwrite: bool = False
) -> None:
    """
    Generate a graph representing the agents performance in an environment according to the specified metric.
    :param agents: the agent names
    :param env: the environment name
    :param seeds: the random seeds over which the average and standard deviation is computed
    :param metric: the name of the metric to draw in the graph
    :param overwrite: True to overwrite the previously computed metric values, False otherwise
    """

    # For each agent, compute the mean and standard deviations of the metric over all seeds.
    ax = None
    summary_statistics = {}
    for agent in agents:

        # Get the path where the summary statistics should be stored.
        relab.initialize(agent, env, paths_only=True)
        summary_statistics_path = join(
            os.environ["STATISTICS_DIRECTORY"], f"{metric}.tsv"
        )

        # Compute the summary statistics for the current agent.
        if exists(summary_statistics_path) and overwrite is False:
            logging.info(
                f"Using already computed summary statistics from: {summary_statistics_path}."
            )
            summary_statistics[agent] = pd.read_csv(summary_statistics_path, sep="\t")
        else:
            logging.info(
                f"Computing summary statistics from TensorBoard files in: {os.environ["TENSORBOARD_DIRECTORY"]}."
            )
            summary_statistics[agent] = compute_summary_statistics(
                agent, env, seeds, metric, summary_statistics_path
            )

        # Draw the mean as a solid curve, and the standard deviation as the shaded area.
        statistics = summary_statistics[agent]
        lower_bound = statistics["mean"] - statistics["std"]
        upper_bound = statistics["mean"] + statistics["std"]
        ax = sns.lineplot(statistics, x="step", y="mean", ax=ax)
        plt.fill_between(
            statistics["step"], lower_bound.values, upper_bound.values, alpha=0.1
        )

    # Set the legend of the figure, and the axis labels with labels sorted in natural order.
    ax.set_title(env)
    ax.legend(handles=ax.lines, labels=agents)
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel(display_name(metric))

    # Save the figure comparing the agents.
    MatPlotLib.save_figure(
        figure_path=join(os.environ["GRAPH_DIRECTORY"], f"{metric}.pdf")
    )


def main():
    """
    Entry point of the draw_graph.py script.
    """

    # Parse the script arguments.
    parser = ArgumentParser(
        prog="draw_graph.py", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["DQN", "RainbowDQN", "RainbowIQN"],
        help="name of the agents whose metric should be added to the graph",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ALE/Pong-v5",
        help="name of the environment for which to draw the graph",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[i for i in range(5)],
        help="random seeds to use",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_episodic_reward",
        help="the metric to display in the graph",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=True,
        help="whether to overwrite the previously computed metric values",
    )
    args = parser.parse_args()

    # Generate a graph representing the agents performance in an environment according to the specified metric.
    draw_graph(
        agents=args.agents,
        env=args.env,
        seeds=args.seeds,
        metric=args.metric,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
