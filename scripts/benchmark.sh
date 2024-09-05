#!/bin/bash

sbatch scripts/bechmark_taxi.sh dylam
sbatch scripts/bechmark_pendulum.sh dylam
sbatch scripts/bechmark_lunarlander.sh dylam
sbatch scripts/bechmark_halfcheetah.sh dylam
sbatch scripts/bechmark_hopper.sh dylam
sbatch scripts/bechmark_humanoid.sh dylam
sbatch scripts/bechmark_vss.sh dylam
sbatch scripts/bechmark_halfcheetahef.sh dylam
sbatch scripts/bechmark_vssef.sh dylam
