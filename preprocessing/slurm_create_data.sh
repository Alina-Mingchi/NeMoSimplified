#!/bin/bash
#SBATCH --time 00:59:00
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --partition cpu
#SBATCH --mem=16G
#SBATCH --account pause

#SBATCH --job-name=create_data
#SBATCH --output=/code/slurm_logs/create-data-%j.out
#SBATCH --error=/code/slurm_logs/create-data-%j.err

eval "$(/idiap/temp/mhou/miniconda3/bin/conda shell.bash hook)"
conda activate nemo

mkdir -p /idiap/temp/mhou/code/slurm_stdout/$SLURM_JOB_ID

echo 'Create '

python3 /idiap/temp/mhou/code/StoRM/preprocessing/create_data_noisetype_name.py --task enh --speech pvqd --noise chime