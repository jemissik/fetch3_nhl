#!/bin/bash
#SBATCH --job-name=main
#SBATCH --time=50:00
#SBATCH --nodes=1 --ntasks-per-node=28
#SBATCH --output=output/boa_main_%j.log
#SBATCH --account=PAS0409

set -x  # for displaying the commands in the log for debugging

config_path=$1
data_path=$2
output_path=$3

echo `pwd`
echo $config_file

source activate fetch3-dev

python main.py --config_path $config_path --data_path $data_path --output_path $output_path

echo "Finished with main.py"