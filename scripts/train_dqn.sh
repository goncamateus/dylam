#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem 32G
#SBATCH -c 16
#SBATCH --gpus=1
#SBATCH -p long
#SBATCH --mail-type=FAIL,END,ARRAY_TASKS
#SBATCH --mail-user=mgm4@cin.ufpe.br

# Load modules and activate python environment
module load Python3.10 Xvfb freeglut glew
source $HOME/doc/c1/bin/activate
which python
cd $HOME/doc/dylam
export MUJOCO_GL=osmesa
# Run the script
python train_dqn.py --env $1 --setup $2 --capture-video --video-freq 20 --track
