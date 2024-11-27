from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from benchmarks import agents
import benchmarks
from benchmarks import environments


def run_demo(agent_name, env_name, seed, checkpoint_index):
    """
    Demonstrate the policy learned by a reinforcement learning agent on a gym environment.
    :param agent_name: the agent name
    :param env_name: the environment name
    :param seed: the random seed
    :param checkpoint_index: the number of training steps corresponding to the checkpoint to load
    """

    # Initialize the benchmark.
    benchmarks.initialize(agent_name, env_name, seed)

    # Create the environment.
    env = environments.make(env_name, render_mode="rgb_array")

    # Create and train the agent.
    agent = agents.make(agent_name, training=False)
    agent.load(f"model_{checkpoint_index}.pt")
    agent.demo(env, f"demo_{checkpoint_index}.gif")


if __name__ == "__main__":

    # Parse the script arguments.
    parser = ArgumentParser(prog="run_demo", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--agent", type=str, default="DQN", help="name of the agent whose policy needs to be demonstrated")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="name of the environment on which to demonstrate the agent's policy")
    parser.add_argument("--seed", type=int, default=0, help="random seed to use")
    parser.add_argument("--index", type=int, default=8500000, help="index of the checkpoint to load")
    args = parser.parse_args()

    # Demonstrate the policy learned by a reinforcement learning agent on a gym environment.
    run_demo(agent_name=args.agent, env_name=args.env, seed=args.seed, checkpoint_index=args.index)
