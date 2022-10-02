#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=72:00:00
#SBATCH --mem=1000GB
#SBATCH --job-name=preprocess

module purge

singularity exec --nv \
	    --overlay /scratch/ab10313/singularity_overlays/overlay-50G-10M.ext3 \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python preprocess_llc4320_save_lat_-57_-42_lon_23_37.py"


