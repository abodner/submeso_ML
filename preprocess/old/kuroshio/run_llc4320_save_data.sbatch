#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=save-llc4320

module purge

singularity exec --nv \
	    --overlay /scratch/ab10313/singularity_overlays/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python -u llc4320_kuroshio.py"
