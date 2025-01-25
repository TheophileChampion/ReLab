## 📗 Tutorial

In this tutorial, we will delve into the nitty gritty of ReLab's python API. 
You will learn about ReLab's configuration, the structure of the data directory, 
as well as the available agents and environments. Finally, by the end of this 
tutorial you should be able to run your first ReLab experiment.

### 1. Understanding the Data Directory Structure  

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

### 2. ReLab Configuration and Initialization

ReLab's configuration allows you to customize key aspects of training and logging. Here are the most relevant entries:

- `max_n_steps`: Maximum number of training iterations (default: 50,000,000).  
  Defines the iterations at which training is stopped.

- `checkpoint_frequency`: Number of training iterations between model checkpoints (default: 500,000).  
  Checkpoints save the agent's state, enabling you to resume training or analyze the agent progress.

- `tensorboard_log_interval`: Number of training iterations between TensorBoard log updates (default: 5,000).  
  Controls how frequently training metrics (e.g., rewards) are logged for visualization.

- `save_all_replay_buffers`: Determines whether all replay buffers are saved (default: `False`).  
  If `False`, only the replay buffer associated with the most recent checkpoint is saved.

Before training, ReLab must be initialized using the `relab.initialize()` function. Here's a quick breakdown:

```python
relab.initialize(
    agent_name="DQN",          # Name of the agent (e.g., "DQN", "RainbowDQN").
    env_name="ALE/Pong-v5",    # Environment on which the agent will train.
    seed=0,                    # Random seed for reproducibility.
    data_directory=None,       # Path for storing all data; defaults to "./data".
    paths_only=False           # If True, initializes paths without setting up the framework.
)
```

This function performs several key steps:  
- Ensures reproducibility by setting the random seed for NumPy, Python, and PyTorch.  
- Registers additional environments (e.g., Atari games and custom environments) with the Gym framework.  
- Creates environment variables such as `CHECKPOINT_DIRECTORY`, and `TENSORBOARD_DIRECTORY` describing where each file should be stored. By default, these directories are organized under `./data`.

### 3. Creating Agents

The `relab.agents.make()` function is a factory method that simplifies the creation of reinforcement learning agents in ReLab. By passing the name of the desired agent and optional keyword arguments, you can create and configure agents with ease.

#### 3.1. Function Overview

```python
def make(agent_name: str, **kwargs: Any) -> AgentInterface:
```

- `agent_name`: The name of the agent to instantiate. Must be one of the supported agents (listed below). If an unsupported name is provided, the function raises an error.  
- `kwargs`: Keyword arguments forwarded to the agent's constructor, allowing you to customize the agent's behavior.

#### 3.2. Example Usage

```python
from relab import agents

# Create a Dueling Double DQN agent.
agent = agents.make("DuelingDDQN", learning_rate=0.0001, gamma=0.99)
```

#### 3.3. Supported Agents: Overview Table

Here’s a table summarizing the supported agents in ReLab. It includes their full names, abbreviations, and key characteristics such as whether they are value-based, distributional, random, or learn a world model.

| **Abbreviation**   | **Full Name**                           | **Value-Based** | **Distributional** | **Random Actions**               | **World Model** |
|---------------------|-----------------------------------------|-----------------|--------------------|----------------------------------|-----------------|
| **DQN**            | Deep Q-Network                         | ✅              | ✖️                 | ✖️                                | ✖️              |
| **DDQN**           | Double Deep Q-Network                  | ✅              | ✖️                 | ✖️                                | ✖️              |
| **CDQN**           | Categorical Deep Q-Network             | ✅              | ✅                 | ✖️                                | ✖️              |
| **MDQN**           | Multi-step Deep Q-Network              | ✅              | ✖️                 | ✖️                                | ✖️              |
| **QRDQN**          | Quantile Regression Deep Q-Network     | ✅              | ✅                 | ✖️                                | ✖️              |
| **NoisyDQN**       | Noisy Deep Q-Network                   | ✅              | ✖️                 | ✖️ (noisy layers for exploration) | ✖️ |
| **NoisyDDQN**      | Noisy Double Deep Q-Network            | ✅              | ✖️                 | ✖️ (noisy layers for exploration) | ✖️ |
| **NoisyCDQN**      | Noisy Categorical Deep Q-Network       | ✅              | ✅                 | ✖️ (noisy layers for exploration) | ✖️ |
| **DuelingDQN**     | Dueling Deep Q-Network                 | ✅              | ✖️                 | ✖️                                | ✖️              |
| **DuelingDDQN**    | Dueling Double Deep Q-Network          | ✅              | ✖️                 | ✖️                                | ✖️              |
| **PrioritizedDQN** | Prioritized Experience Replay DQN      | ✅              | ✖️                 | ✖️                                | ✖️              |
| **PrioritizedDDQN**| Prioritized Experience Replay DDQN     | ✅              | ✖️                 | ✖️                                | ✖️              |
| **PrioritizedMDQN**| Prioritized Multi-step DQN             | ✅              | ✖️                 | ✖️                                | ✖️              |
| **RainbowDQN**     | Rainbow Deep Q-Network                 | ✅              | ✅                 | ✖️                                | ✖️              |
| **RainbowIQN**     | Rainbow with Implicit Quantile Network | ✅              | ✅                 | ✖️                                | ✖️              |
| **IQN**            | Implicit Quantile Network              | ✅              | ✅                 | ✖️                                | ✖️              |
| **Random**         | Random Agent                           | ✖️              | ✖️                 | ✅                                | ✖️              |
| **VAE**            | Variational Autoencoder                | ✖️              | ✖️                 | ✅                                | ✅              |
| **BetaVAE**        | Beta Variational Autoencoder           | ✖️              | ✖️                 | ✅                                | ✅              |
| **DiscreteVAE**    | Discrete Variational Autoencoder       | ✖️              | ✖️                 | ✅                                | ✅              |
| **JointVAE**       | Joint Variational Autoencoder          | ✖️              | ✖️                 | ✅                                | ✅              |
| **HMM**            | Hidden Markov Model                   | ✖️              | ✖️                 | ✅                                | ✅              |
| **BetaHMM**        | Beta Hidden Markov Model               | ✖️              | ✖️                 | ✅                                | ✅              |
| **DiscreteHMM**    | Discrete Hidden Markov Model           | ✖️              | ✖️                 | ✅                                | ✅              |
| **JointHMM**       | Joint Hidden Markov Model              | ✖️              | ✖️                 | ✅                                | ✅              |

##### Notes:
1. **Value-Based Agents**: Agents like DQN and DDQN focus on learning a value function to determine optimal actions.
2. **Distributional Agents**: Distributional RL agents (e.g., QRDQN, CDQN) model the distribution of returns instead of estimating a single expected return.
3. **Random Actions**: Several agents take random actions, they can be used either to learn a world model or as a baseline for comparing more sophisticated agents.
4. **World Model Agents**: Agents like VAEs and HMMs focus on learning a representation of the environment or the "world model," which can be used for planning or analysis.

### 4. Creating Environments

// TODO

### 5. Training your First Agent

// TODO

### 6. Running your First Experiment

// TODO
