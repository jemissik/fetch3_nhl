#!/bin/bash
#SBATCH --job-name=boa_run
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks-per-node=28
#SBATCH --output=%j.log
#SBATCH --account=PAS0409

set -x

config_path=$1

echo 'starting optimization'

python -m boa --config-path $config_path

echo 'completed optimization'