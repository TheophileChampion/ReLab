from benchmarks import agents
import benchmarks
from benchmarks import environments


def run_demo(agent_name, env_name, seed, checkpoint_index):
    """
    Demonstrate the policy learnt by a reinforcement learning agent on a gym environment.
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

    # Demonstrate the policy learnt by a reinforcement learning agent on a gym environment.
    run_demo(agent_name="DuelingDDQN", env_name="ALE/Pong-v5", seed=0, checkpoint_index=8500000)
