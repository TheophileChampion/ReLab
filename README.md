# ![ReLab](./relab-logo.png)

---

ReLab is a powerful and user-friendly library designed to streamline
reinforcement learning experiments. With a curated collection of well-known
RL agents and seamless integration with Gym environments, ReLab empowers you
to train agents effortlessly, visualize their learned behaviors, and compare
their performance across diverse metrics. Whether you're a researcher exploring
state-of-the-art algorithms or a practitioner fine-tuning agents for real-world
applications, ReLab simplifies the process, saving you time and effort.

<!-- toc -->

- [Installation](#installation)
  - [Install Poetry](#install-poetry)
  - [Install ReLab and its dependencies](#install-relab-and-its-dependencies)
- [ReLab Fundamentals](#relab-fundamentals)
- [Documentation](#documentation)
- [Releases and Contributing](#releases-and-contributing)
- [License](#license)
<!-- tocstop -->

## Installation

### Install Poetry

Linux, macOS, Windows (WSL):
```console
curl -sSL https://install.python-poetry.org | python3 -
```

Windows (Powershell)
```console
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Install ReLab and its dependencies

Clone the ReLab repository:
```console
git clone git@github.com:TheophileChampion/ReLab.git
cd ReLab
```

Install ReLab's package and its dependencies inside a virtual environment:
```console
poetry install
```

Check that the installation was successful:
```console
poetry run test_install
```

## ReLab Fundamentals

ReLab provides a collection of well-known reinforcement learning agents and enables you to train them on any [Gym](https://gymnasium.farama.org/) environment.
You can then visualize the learned policies and compare the performance of various agents.
Before proceeding with the following sections, make sure that ReLab is [installed](#installation) and activate the virtual environment using the following command:
```console
source ./.venv/bin/activate
```

### Training an Agent

An agent can be trained by running the following command:

```console
poetry run training --agent DQN --env ALE/Pong-v5 --seed 0
```

The training script accepts three parameters:

- `--agent` specifies the reinforcement learning agent to train,
- `--env` defines the environment in which the agent will be trained, and
- `--seed` sets the random seed to ensure reproducibility.

As the agent is learning, you can visualize its performance using the following command:

```console
tensorboard --logdir=./data/runs/
```

<!-- markdown-link-check-disable -->
This will open a graphical interface at <http://localhost:6006/>, allowing you to monitor:
<!-- markdown-link-check-enable -->

- the agent's mean episodic reward,
- the training speed (in milliseconds per training iteration), and
- the memory usage of the training script.

### Visualizing a Learned Policy

By default, ReLab saves the learned policy every 500,000 training iterations.
Once an agent has been trained, you can visualize its learned policy using the following command:

```console
poetry run demo --agent DQN --env ALE/Pong-v5 --seed 0
```

These parameters should look familiar, as they are identical to those used in the training script.
By default, ReLab demonstrates the latest policy.
However, you can specify a particular model checkpoint using the following command:

```console
poetry run demo --agent DQN --env ALE/Pong-v5 --seed 0 --index 1000000
```

Here, the `--index` parameter allows you to select the policy learned by the agent after 1,000,000 training iterations.
After running the above command, ReLab will generate a GIF of the agent's behavior, which can be found in:

```console
./data/demos/Pong-v5/DQN/0/demo_1000000.gif
```

<img alt="" width="175" src="./assets/demo_prioritized_ddqn_pong.gif">

### Comparing the Performance of Various Agents

ReLab also provides a script to generate graphs summarizing the agent performance:

```console
poetry run draw_graph --agents DQN --env ALE/Pong-v5 --seeds 0 --metric mean_episodic_reward
```

Importantly, the parameters `--agents` and `--seeds` are now plural because the script accepts a list of agents and seeds.
The `--metric` parameter allows you to compare agents based on various metrics, such as episodic mean reward, training speed, or memory usage.
This script can also be used to compare multiple agents across different seeds, as shown below:

```console
poetry run draw_graph --agents DQN RainbowDQN --env ALE/Pong-v5 --seeds 0 1 2 3 4 --metric mean_episodic_reward
```

When multiple seeds are provided, the graph will display a solid line representing the average metric, along with a shaded area indicating the metric's standard deviation.

<img alt="" src="./assets/mean_episodic_reward.png" width="500"/>

For a deeper dive into advanced use cases of ReLab and additional examples, check out our [In-Depth Tutorial](./TUTORIAL.md).

## Documentation

ReLab [documentation](https://theophilechampion.github.io/ReLab/) is based on doxygen and hosted with GitHub Pages.

## Releases and Contributing

Please let us know if you encounter a bug by [filing an issue](https://github.com/TheophileChampion/ReLab/issues).

ReLab follows a "fork and pull request" workflow. If you are planning to contribute, you must:

- Fork ReLab to your GitHub account
- Clone your fork on your computer
- Create a new branch to work on
- Make changes and commit them with a clear description
- Push your branch to your forked GitHub repository
- Open a pull request to the ReLab project

ReLab is still in beta. The latest version of ReLab is version 1.0.0-b.

## License

ReLab has a MIT license, as found in the [license](https://github.com/TheophileChampion/ReLab/blob/main/LICENSE.md) file.
