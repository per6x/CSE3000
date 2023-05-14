#!/bin/bash

#SBATCH --job-name=rf_grid_search
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=8G
#SBATCH --account=education-eemcs-courses-cse3000

# Load modules:
module load 2022r2
module load openmpi
module load python
module load py-scikit-learn
module load py-matplotlib
module load py-numpy
module load py-pip

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python /tudelft.net/staff-umbrella/ppersianovcse3000/CSE3000/kraken_taxonomy/dhcp_rf_grid.py
