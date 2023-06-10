#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --account=education-eemcs-courses-cse3000

# Load modules:
module load 2023r1
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate rp
cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
conda deactivate

