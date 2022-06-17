#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --time=60:00
#SBATCH --nodes=1 --ntasks-per-node=12
#SBATCH --output=logs/%j.log
#SBATCH --account=PAS0409

set -x  # for displaying the commands in the log for debugging

config_dir=$SLURM_SUBMIT_DIR/opt_umbs_M8.yml


echo $config_dir

python ~/repos/fetch3_nhl/optimize/run_optimization.py --config_path $config_dir

echo "Finished with run_optimization.py"