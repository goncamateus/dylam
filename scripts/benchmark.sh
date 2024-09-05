#!/bin/bash

sbatch scripts/benchmark_taxi.sh c1
sbatch scripts/benchmark_pendulum.sh c1
sbatch scripts/benchmark_lunarlander.sh c1
sbatch scripts/benchmark_halfcheetah.sh c1
sbatch scripts/benchmark_hopper.sh c1
sbatch scripts/benchmark_humanoid.sh c1
sbatch scripts/benchmark_vss.sh c1
sbatch scripts/benchmark_halfcheetahef.sh c1
sbatch scripts/benchmark_vssef.sh c1
