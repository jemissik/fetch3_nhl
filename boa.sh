#!/bin/bash
#SBATCH --job-name=boa
#SBATCH --time=2:00:00
#SBATCH --nodes=1 --ntasks-per-node=14
#SBATCH --output=output/boa_%j.log
#SBATCH --account=PAS0409

set -x  # for displaying the commands in the log for debugging

config_file=$1

echo `pwd`
echo $config_file

source activate fetch3-dev

python optimization_run.py --config_file $config_file -b

echo "Finished with run_optimization.py"