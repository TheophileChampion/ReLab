#!/bin/bash
# This script should be used to install the project dependencies.
# After executing this script, activate the virtual environment using:
#     source ./venv-benchmarks/bin/activate
# Launch a dummy script such as:
#     python3 ./scripts/run_training.py --agent DQN --env "ALE/Pong-v5" --seed 0

# Create the virtual environment and install all the dependencies.
python3.12 -m venv venv-benchmarks
source ./venv-benchmarks/bin/activate
pip install -r requirements.txt
