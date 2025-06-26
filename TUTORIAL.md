# 📗 Tutorial

ReLab is a versatile and powerful library for training, evaluating, and analyzing
reinforcement learning agents. This tutorial will walk you through its core features,
including creating environments, defining agents, and training your first model using
ReLab’s Python API. Additionally, you'll learn how to run complete experiments using
ReLab’s command-line interface.

## 1. Understanding the Data Directory Structure

When running ReLab scripts, the library organizes all generated files into
a `data` directory. This structured directory ensures that your experiment
outputs are logically grouped, making it easy to access and analyze the
results. Below is an overview of the `data` directory and its purpose:

```commandline
data/
├── demos
│   └── <Environment>
│        └── <Agent>
│             └── <Seed>
│                  └── demo_<iteration>.gif
├── graphs
│   └── <Environment>
│        ├── <Metric>.pdf
│        └── <Agent>
│             └── <Metric>.tsv
├── runs
│   └── <Environment>
│        └── <Agent>
│             └── <Seed>
│                  └── events.out.tfevents.<timestamp>.<hostname>.<PID>.<UID>
└── saves
    └── <Environment>
         └── <Agent>
              └── <Seed>
                   ├── buffer.pt
                   └── model_<iteration>.pt
```

Here’s what each folder contains:

1. `demos/`:  
   This folder contains GIFs demonstrating the agent's learned policy.

   - For each environment, agent, and random seed, ReLab generates GIFs representing specific training iterations.
   - Example: `demo_500000.gif` shows the agent's behavior after 500,000 training iterations.

2. `graphs/`:  
   This folder contains visualizations of agent performance metrics.

   - Metric graphs (e.g., `mean_episodic_reward.pdf`) are stored for each environment and summarize the performance of one or more agents.
   - Detailed data files (e.g., `mean_episodic_reward.tsv`) are also stored here for individual agents, containing the mean and standard deviation of the specified metric at each training step.

3. `runs/`:  
   This folder logs training data in a format compatible with [TensorBoard](https://www.tensorflow.org/tensorboard).

   - Each environment-agent-seed combination has its own folder containing event files (e.g., `events.out.tfevents...`) that allow you to track the agent’s progress during training.

4. `saves/`:  
   This folder stores the saved models for each training session.
   - Model checkpoints are saved for specific training iterations (e.g., `model_500000.pt`), allowing you to reload and evaluate the agent at different stages of training.
   - Replay buffer checkpoint (e.g., `buffer.pt`) saves the replay buffer associated with the last checkpoint iteration, ensuring training can resume seamlessly from where it was left off. For example, if the directory contains `model_500.pt` and `model_1000.pt`, then `buffer.pt` corresponds to the replay buffer at iteration 1000.

By organizing experiment outputs in this way, ReLab ensures that your data is easy to locate and manage, enabling you to efficiently analyze results, compare agents, and showcase their learned behaviors.

## 2. ReLab Configuration and Initialization

ReLab's configuration allows you to customize key aspects of training and logging. Here are the most relevant entries:

- `max_n_steps`: Maximum number of training iterations (default: 50,000,000).  
  Defines the iterations at which training is stopped.

- `checkpoint_frequency`: Number of training iterations between model checkpoints (default: 500,000).  
  Checkpoints save the agent's state, enabling you to resume training or analyze the agent progress.

- `tensorboard_log_interval`: Number of training iterations between TensorBoard log updates (default: 5,000).  
  Controls how frequently training metrics (e.g., rewards) are logged for visualization.

- `save_all_replay_buffers`: Determines whether all replay buffers are saved (default: `False`).  
  If `False`, only the replay buffer associated with the most recent checkpoint is saved.

**Example Usage**

```python
# Retrieve a specific config value.
max_steps = relab.config("max_n_steps")
print(f"Maximum training steps: {max_steps}")
```

---

Before doing anything with ReLab, the `relab.initialize()` function must be called.
It is the first step to setting up the library, ensuring that all paths are properly configured.
Here's a quick breakdown:

```python
relab.initialize(
    agent_name="DQN",          # Name of the agent (e.g., "DQN", "RainbowDQN").
    env_name="ALE/Pong-v5",    # Environment on which the agent will be trained or evaluated.
    seed=0,                    # Random seed for reproducibility.
    data_directory=None,       # Path for storing all data; defaults to "./data".
    paths_only=False           # If True, initializes paths without setting up the framework.
)
```

This function performs several key steps:

- Ensures reproducibility by setting the random seed for NumPy, Python, and PyTorch.
- Registers additional environments (e.g., Atari games and custom environments) with the Gym framework.
- Initializes environment variables (e.g., `CHECKPOINT_DIRECTORY` and `TENSORBOARD_DIRECTORY`) to define where specific files are stored, ensuring consistency across scripts.

## 3. Creating Agents

The `relab.agents.make()` function is a factory method that simplifies the creation of reinforcement learning agents in ReLab. By passing the name of the desired agent and optional keyword arguments, you can create and configure agents with ease.

### 3.1. Function Overview

```python
def make(agent_name: str, **kwargs: Any) -> AgentInterface:
```

- `agent_name`: The name of the agent to instantiate. Must be one of the supported agents (listed below). If an unsupported name is provided, the function raises an error.
- `kwargs`: Keyword arguments forwarded to the agent's constructor, allowing you to customize the agent's behavior.

**Example Usage**

```python
from relab import agents

# Create a Dueling Double DQN agent.
agent = agents.make("DuelingDDQN", learning_rate=0.0001, gamma=0.99)
```

### 3.2. Supported Agents: Overview Table

Here’s a table summarizing the supported agents in ReLab. It includes their full names, abbreviations, and key characteristics such as whether they are value-based, distributional, random, or learn a world model.

| **Abbreviation**    | **Full Name**                          | **Value-Based** | **Distributional** | **Random Actions**                | **World Model** |
|---------------------|----------------------------------------|-----------------|--------------------|-----------------------------------|-----------------|
| **DQN**             | Deep Q-Network                         | ✅               | ✖️                 | ✖️                                | ✖️              |
| **DDQN**            | Double Deep Q-Network                  | ✅               | ✖️                 | ✖️                                | ✖️              |
| **CDQN**            | Categorical Deep Q-Network             | ✅               | ✅                  | ✖️                                | ✖️              |
| **MDQN**            | Multi-step Deep Q-Network              | ✅               | ✖️                 | ✖️                                | ✖️              |
| **QRDQN**           | Quantile Regression Deep Q-Network     | ✅               | ✅                  | ✖️                                | ✖️              |
| **NoisyDQN**        | Noisy Deep Q-Network                   | ✅               | ✖️                 | ✖️ (noisy layers for exploration) | ✖️              |
| **NoisyDDQN**       | Noisy Double Deep Q-Network            | ✅               | ✖️                 | ✖️ (noisy layers for exploration) | ✖️              |
| **NoisyCDQN**       | Noisy Categorical Deep Q-Network       | ✅               | ✅                  | ✖️ (noisy layers for exploration) | ✖️              |
| **DuelingDQN**      | Dueling Deep Q-Network                 | ✅               | ✖️                 | ✖️                                | ✖️              |
| **DuelingDDQN**     | Dueling Double Deep Q-Network          | ✅               | ✖️                 | ✖️                                | ✖️              |
| **PrioritizedDQN**  | Prioritized Experience Replay DQN      | ✅               | ✖️                 | ✖️                                | ✖️              |
| **PrioritizedDDQN** | Prioritized Experience Replay DDQN     | ✅               | ✖️                 | ✖️                                | ✖️              |
| **PrioritizedMDQN** | Prioritized Multi-step DQN             | ✅               | ✖️                 | ✖️                                | ✖️              |
| **RainbowDQN**      | Rainbow Deep Q-Network                 | ✅               | ✅                  | ✖️                                | ✖️              |
| **RainbowIQN**      | Rainbow with Implicit Quantile Network | ✅               | ✅                  | ✖️                                | ✖️              |
| **IQN**             | Implicit Quantile Network              | ✅               | ✅                  | ✖️                                | ✖️              |
| **Random**          | Random Agent                           | ✖️              | ✖️                 | ✅                                 | ✖️              |
| **VAE**             | Variational Autoencoder                | ✖️              | ✖️                 | ✅                                 | ✅               |
| **BetaVAE**         | Beta Variational Autoencoder           | ✖️              | ✖️                 | ✅                                 | ✅               |
| **HMM**             | Hidden Markov Model                    | ✖️              | ✖️                 | ✅                                 | ✅               |
| **BetaHMM**         | Beta Hidden Markov Model               | ✖️              | ✖️                 | ✅                                 | ✅               |
| **CHMM**            | Critical Hidden Markov Model           | ✅               | ✖️                 | ✖️                                | ✅               |

**Notes:**

1. **Value-Based Agents**: Agents like DQN and DDQN focus on learning a value function to determine optimal actions.
2. **Distributional Agents**: Distributional RL agents (e.g., QRDQN, CDQN) model the distribution of returns instead of estimating a single expected return.
3. **Random Actions**: Several agents take random actions, they can be used either to learn a world model or as a baseline for comparing more sophisticated agents.
4. **World Model Agents**: Agents like VAEs and HMMs focus on learning a representation of the environment or the "world model," which can be used for planning or analysis.

## 4. Creating Environments

The `relab.environments.make()` function is a factory that provides an easy and customizable way to set up
Gym environments for training reinforcement learning agents.

### 4.1. Function Overview

```python
def make(env_name: str, **kwargs: Any) -> Env:
```

- `env_name`: The name of the environment to instantiate.
- `kwargs`: Keyword arguments forwarded to the environment's constructor, allowing you to customize the environment.

The function applies several preprocessing steps:

- **Environment Setup**: Initializes the environment with `gym.make`, by default the entire action space is used (18 actions for all Atari games).
- **FireReset Wrapper**: Ensures that the environment resets properly by simulating a fire action where applicable.
- **Atari Preprocessing**:
  - Rescales observations to the configured screen size (`screen_size`).
  - Converts observations to grayscale.
  - Scales pixel values for improved learning stability.
  - Skips frames as defined in `frame_skip`.
- **Frame Stacking**: Stacks the last `stack_size` observations to provide temporal context for agents.
- **Torch Integration**: Converts environment outputs to PyTorch tensors for seamless agent interaction.

**Example Usage**

```python
from relab import environments

# Create an environment running the Atari game pong.
env = environments.make("ALE/Pong-v5")
```

### 4.2. Predefined Atari Game Sets

At times, you might want to evaluate your agents on a specific subset of Atari games.
ReLab provides three predefined Atari benchmarks to simplify this process:

1. `small_benchmark_atari_games()`

   - Returns a small subset of five Atari games for quick benchmarking:
     - Breakout
     - Freeway
     - Ms. Pac-Man
     - Pong
     - Space Invaders

2. `benchmark_atari_games()`

   - Returns the standard set of 57 Atari games used in reinforcement learning research benchmarks.
   - Includes all games from `small_benchmark_atari_games()` plus additional titles like Asteroids, Seaquest, and Montezuma’s Revenge.

3. `all_atari_games()`
   - Returns all available Atari games, including the benchmark games and extra titles like Adventure and Air Raid.

**Example Usage:**

```python
from relab import environments

# Retrieve the list of Atari benchmark games.
benchmark_games = environments.atari_benchmark()
print(f"Total Atari Benchmark Games: {len(benchmark_games)}")
```

## 5. Training your First Agent

By now, you’ve learned about ReLab's features, how to configure the library, create agents and environments,
and manage saved data and benchmarks. Let’s bring it all together with a complete training script to
demonstrate how these components work in practice:

```python
from relab import agents, environments
import relab


def run_training(agent: str, env: str, seed: int) -> None:
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


if __name__ == "__main__":
    # Train a reinforcement learning agent on a gym environment.
    run_training(agent="DDQN", env="ALE/Pong-v5", seed=0)
```

## 6. Running your First Experiment

While you could use Poetry to train and demonstrate the policy of individual agents,
ReLab enables you to run full-scale experiments. An experiment automates training,
evaluation, and result visualization across multiple agents, environments, and
random seeds. Here’s a breakdown of what the script does:

1. **Training Agents**: For each combination of agent, environment, and seed, the script launches training jobs either locally or using Slurm (a workload manager for distributed systems).

2. **Policy Demonstrations**: After training, it generates GIFs to visually demonstrate the learned policies for each agent-environment-seed combination.

3. **Performance Analysis**: The script creates performance graphs (e.g., mean episodic rewards with standard deviations) for each environment, summarizing how all agents performed.

4. **Parallelization**: Jobs are managed efficiently either on the local machine (with multiple workers) or on a Slurm cluster, depending on the user’s choice.

**Example Usage:**

- Specify agents, environments, and seeds using command-line arguments. For example:
  ```bash
  poetry run experiment --agents DQN RainbowDQN --envs ALE/Pong-v5 --seeds 0 1 2
  ```
- Use the `--no-local` flag to run experiments using Slurm. Omitting it defaults to run locally.

This script ensures a streamlined workflow for conducting experiments, from training to visualization,
with minimal manual intervention!

## 7. What's Next?

For more details, you can explore the official documentation, which provides an in-depth explanation
of all ReLab’s classes. Additionally, the Python scripts in the `scripts` directory offer practical
examples to help you understand how ReLab works. These resources are great starting points for
deepening your understanding and making the most out of ReLab!
