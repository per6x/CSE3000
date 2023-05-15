#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --partition=compute
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --account=education-eemcs-courses-cse3000

# Load modules:
module load 2022r2
module load openmpi
module load miniconda3
module load python
module load py-numpy py-pip py-scipy py-scikit-learn py-matplotlib



# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate jupyterlab
cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
conda deactivate

