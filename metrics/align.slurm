#!/bin/bash
#SBATCH --job-name=job        # create a short name for your job
#SBATCH -p cryoem
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

ref=$1
vol=$2
outvol=$3
tmpfile=$4

/scratch/gpfs/mj7341/chimerax-1.6.1/bin/ChimeraX --nogui --script "align.py $ref $vol -o $outvol -f $tmpfile" > $tmpfile