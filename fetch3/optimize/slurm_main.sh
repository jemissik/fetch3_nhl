#!/bin/bash
#SBATCH --job-name=%j
#SBATCH --time=7:50:00
#SBATCH --nodes=1 --ntasks-per-node=28
#SBATCH --output=output/%j.log
#SBATCH --account=PAS0409

set -x  # for displaying the commands in the log for debugging

config_path=$1
data_path=$2
output_path=$3
working_dir=$SLURM_SUBMIT_DIR

echo $config_file

source activate fetch3-dev

cd $working_dir

python python main.py --config_path $config_path --data_path $data_path --output_path $output_path

echo "Finished with main.py"