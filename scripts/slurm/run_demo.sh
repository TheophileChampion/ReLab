#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL
#SBATCH --account=bowmanh-theophile-work
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --qos=bbgpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a30:1
# The above parameter can take the following values:
# - gpu:a30:1
# - gpu:a100:1
# - gpu:a100_80:1

set -e

# Load BlueBEAR modules.
module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

# Create the virtual environments directory.
export PROJECT_PATH=/rds/projects/b/bowmanh-theophile-work/ReinforcementLearningBenckmarks
mkdir -p ${PROJECT_PATH}

# Export the path to the node specific environment.
export VENV_DIR="${PROJECT_PATH}/rl-benchmarks-${BB_CPU}"

# Check if virtual environment exists and create it if not.
if [[ ! -d ${VENV_DIR} ]]; then

    # Let the user know that the environment was not found and will be created.
    echo "${VENV_DIR} does not exists, it will be created."

    # Create the virtual environment.
    python3 -m venv --system-site-packages ${VENV_DIR}

    # Activate the environment.
    source ${VENV_DIR}/bin/activate

    # Install project dependencies.
    pip install -r requirements.txt

else

    # Let the user know that the environment was not found and will be activated.
    echo "${VENV_DIR} exists, it will be activated."

    # Activate the environment.
    source ${VENV_DIR}/bin/activate
fi

# Export data directory.
export DATA_DIRECTORY=${PROJECT_PATH}/data/

# Run the script creating a GIF file demonstrating the learned policy corresponding to the provided parameters.
python3 ${PROJECT_PATH}/scripts/run_demo.py $*
