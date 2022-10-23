#!/bin/bash
  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=testfig

module purge

singularity exec --nv \
            --overlay /scratch/ab10313/singularity_overlays/overlay-50G-10M.ext3 \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python testfig.py"
