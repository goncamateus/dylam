#!/bin/bash
#SBATCH --mem 8G
#SBATCH -c 8
#SBATCH --gpus 1
#SBATCH -p short-simple
#SBATCH --mail-type=FAIL,END,ARRAY_TASKS
#SBATCH --mail-user=mgm4@cin.ufpe.br

# Load modules and activate python environment
module load Python3.10
export UV_CACHE_DIR=/tmp/uv-cache/
PROJECT_DIR=$(pwd)
TMP_DIR=/tmp/$USER/dylam/
mkdir -p $TMP_DIR

echo "Copying project files to $TMP_DIR"
rsync -av --exclude='.venv' --exclude='.git/' --exclude='__pycache__/' --exclude='scripts/models/' --exclude='scripts/videos/' --exclude='scripts/wandb/' $PROJECT_DIR/ $TMP_DIR/

echo "Changing to temporary directory $TMP_DIR"
cd $TMP_DIR

echo "Creating Python virtual environment"
uv venv --clear --no-cache --link-mode hardlink

echo "Installing project dependencies"
uv sync --frozen
uv pip install -e .

echo "Starting Job"
uv run --directory scripts/ python train.py --env $1 --setup $2 --track --capture-video

echo "Copying results back to project directory"
rsync -av --exclude='.venv' --exclude='.git/' --exclude='__pycache__/' --exclude='scripts/models/' --exclude='scripts/videos/' --exclude='scripts/wandb/' $TMP_DIR/ $PROJECT_DIR/ 