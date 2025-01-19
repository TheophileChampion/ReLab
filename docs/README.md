## ReLab
## Installation

### Clone the ReLab source code

```
git clone git@github.com:TheophileChampion/ReLab.git
cd ReLab
```

### Download ReLab dependencies

```
python3.12 -m venv venv-relab
source ./venv-relab/bin/activate
pip install -r requirements.txt
```

### Ensure that python scripts can find the ReLab package

```
pwd > ./venv-relab/lib/python3.12/site-packages/relab.pth
```

### Check that the installation was successful

```
python ./scripts/test_install.py
```

Note, when running the above command for the first time, ReLab will compile its C++ library.

## Getting Started

ReLab provides a collection of well-known reinforcement learning agents and enables you to train them on any [Gym](https://gymnasium.farama.org/) environment.
You can then visualize the learned policies and compare the performance of various agents. 
Before proceeding with the following sections, make sure that ReLab is [installed](#installation) and activate the virtual environment using the following command:
```
source ./venv-relab/bin/activate
```

### Training an Agent

An agent can be trained by running the following command:
```
python ./scripts/run_training.py --agent DQN --env ALE/Pong-v5 --seed 0
```
The training script accepts three parameters:
- `--agent` specifies the reinforcement learning agent to train,
- `--env` defines the environment in which the agent will be trained, and
- `--seed` sets the random seed to ensure reproducibility.

As the agent is learning, you can visualize its performance using the following command:
```
tensorboard --logdir=./data/runs/
```
This will open a graphical interface at http://localhost:6006/, allowing you to monitor:
- the agent's mean episodic reward,
- the training speed (in milliseconds per training iteration), and
- the memory usage of the training script.

### Visualizing a Learned Policy

By default, ReLab saves the learned policy every 500,000 training iterations.
Once an agent has been trained, you can visualize its learned policy using the following command:
```
python ./scripts/run_demo.py --agent DQN --env ALE/Pong-v5 --seed 0
```
These parameters should look familiar, as they are identical to those used in the training script.
By default, ReLab demonstrates the latest policy.
However, you can specify a particular model checkpoint using the following command:
```
python ./scripts/run_demo.py --agent DQN --env ALE/Pong-v5 --seed 0 --index 1000000
```
Here, the `--index` parameter allows you to select the policy learned by the agent after 1,000,000 training iterations.
After running the above command, ReLab will generate a GIF of the agent's behavior, which can be found in:
```
./data/demos/Pong-v5/DQN/0/demo_1000000.gif
```

// TODO add GIF demonstrating the learned policy on various games

### Comparing the Performance of Various Agents

ReLab also provides a script to generate graphs summarizing the agent performance:
```
draw_graph --agents DQN --env ALE/Pong-v5 --seeds 0 --metric mean_episodic_reward
```
Importantly, the parameters `--agents` and `--seeds` are now plural because the script accepts a list of agents and seeds.
The `--metric` parameter allows you to compare agents based on various metrics, such as episodic mean reward, training speed, or memory usage.
This script can also be used to compare multiple agents across different seeds, as shown below:
```
draw_graph --agents DQN RainbowDQN --env ALE/Pong-v5 --seeds 0 1 2 3 4 --metric mean_episodic_reward
```
When multiple seeds are provided, the graph will display a solid line representing the average metric, along with a shaded area indicating the metric's standard deviation.

// TODO add figures comparing the agents performances

For a deeper dive into advanced use cases of ReLab and additional examples, check out our [In-Depth Tutorial](TUTORIAL.md).

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

ReLab has a MIT license, as found in the [LICENSE](LICENSE) file.
