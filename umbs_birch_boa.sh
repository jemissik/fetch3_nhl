#!/bin/bash
#SBATCH --job-name=boa
#SBATCH --time=7:50:00
#SBATCH --nodes=1 --ntasks-per-node=14
#SBATCH --output=output/boa_%j.log
#SBATCH --account=PAS0409

set -x  # for displaying the commands in the log for debugging

config_file=$SLURM_SUBMIT_DIR/config_files/opt_umbs_birch_avg.yml


echo $config_file

source activate fetch3-dev

cd ~/fetch3_nhl

python optimization_run.py --config_file $config_file -b

echo "Finished with run_optimization.py"