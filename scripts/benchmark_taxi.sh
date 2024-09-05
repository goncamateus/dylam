#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH -c 8
#SBATCH -p long
#SBATCH --mail-type=FAIL,END,ARRAY_TASKS
#SBATCH --mail-user=mgm4@cin.ufpe.br

# Load modules and activate python environment
module load Python3.10 Xvfb freeglut glew
source $HOME/doc/$1/bin/activate
which python
cd $HOME/doc/dylam

for method in "baseline" "drq" "dylam"
do
    for i in {1..5}:
    do
        python train_q_learning.py --env taxi --setup ${method} --capture-video --video-freq 49 --track
    done
done