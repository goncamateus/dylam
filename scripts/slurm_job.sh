#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH -p short-simple
#SBATCH --account=hansenclever_de_franca_bassani_group
#SBATCH --mail-type=FAIL,END,ARRAY_TASKS
#SBATCH --mail-user=mgm4@cin.ufpe.br

# Load modules and activate python environment
module use /opt/easybuild/modules/all
module load Python3.10 SWIG
export UV_CACHE_DIR=/tmp/uv-cache/
cd ..
PROJECT_DIR=$(pwd)
TMP_DIR=/tmp/$USER/$RANDOM
rm -rf $TMP_DIR
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
for i in {1..5}; do
    (
        sleep $(( (i-1) * 5 ))
        uv run --directory scripts/ python $1 --env $2 --setup $3 --track --capture-video
    ) &
done
wait

echo "Copying results back to project directory"
rsync -av --exclude='.venv' --exclude='.git/' --exclude='__pycache__/' --exclude='scripts/models/' --exclude='scripts/videos/' --exclude='scripts/wandb/' $TMP_DIR/ $PROJECT_DIR/ 
cd /tmp
rm -rf $TMP_DIR