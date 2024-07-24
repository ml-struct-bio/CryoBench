#!/usr/bin/bash
## This file is called "myproj.sbatch"
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --constraint=a100  # if you want a particular type of GPU
#SBATCH --time=72:00:00

module load python/3.9
module load cuda
module load cudnn
#pip install jaxlib
#source ~/virtual_envs/cryosbi_env/bin/activate

python cal_neighb_hit_werror.py > dim_embd 

