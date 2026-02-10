#!/bin/bash

#Submit this script with: sbatch myjob.slurm

#SBATCH --time=96:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=1   # number of CPU cores per task
#SBATCH --partition=standard   # partition
#SBATCH --mem=750G   # memory
#SBATCH -J "process open alex"   # job name
#SBATCH --constraint=rivanna   # cluster name afton or rivanna
#SBATCH --account=sds_ag   # allocation name


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load miniforge
pip3 install --upgrade scipy numpy scikit-learn pandas numba boto3
pip3 install pyscisci

python3 processoa.py