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
source $HOME/doc/$1/bin/activate
which python
cd $HOME/doc/dylam
export MUJOCO_GL=osmesa
for method in "baseline" "dylam"
do
    for i in {1..5}:
    do
        python train.py --env humanoid --setup ${method} --capture-video --video-freq 49 --track
    done
done