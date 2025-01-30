#!/bin/bash
# This script is only useful for running the code on BlueBear.
# After executing this script, activate the virtual environment using:
#     source ./venv-relab/bin/activate
# Launch a dummy script such as:
#     python3 ./scripts/run_experiment --agents DQN --envs "ALE/Pong-v5" --seeds 0
# When the slurm-xxxxxx.out indicates that the virtual environment installation is finished on the node, cancel the job using:
#     scancel -u user_name
# The full experiment can now be launched without conflicts using:
#     python3 ./scripts/run_experiment --agents DQN --envs "ALE/Pong-v5" --seeds 0

# Export environment variables.
ROOT_PATH=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${ROOT_PATH}"
export CONDA_ENV_PATH=/rds/projects/b/bowmanh-theophile-work/deep-active-inference/data/${USER}_dai_env
export CONDA_PKGS_DIRS="/scratch/${USER}/conda"
export TMPDIR=/scratch/${USER}/tmp
export PIP_CACHE_DIR=/scratch/${USER}/pip

# Create the temporary directory if it does not currently exist.
mkdir -p "$TMPDIR"

# Load required models.
module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

# Create the virtual environment and install all the dependencies.
python3 -m venv venv-relab
source ./venv-relab/bin/activate
pip install -r requirements.txt
