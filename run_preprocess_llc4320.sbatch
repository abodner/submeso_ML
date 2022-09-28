#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=preprocess

module purge

singularity exec --nv \
	    --overlay /scratch/ab10313/singularity_overlays/overlay-50G-10M.ext3 \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python preprocess_llc4320.py"


