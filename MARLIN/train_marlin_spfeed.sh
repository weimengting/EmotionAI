#!/bin/bash

#SBATCH --account=project_462000771
#SBATCH --partition=dev-g           # Partition name
#SBATCH --job-name=job_name         # Job name
#SBATCH --output=%x_%j.out       # Output file
#SBATCH --error=%x_%j.err        # Error file
#SBATCH --time=00:30:00                # Time limit (hh:mm:ss)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=2              # Number of CPU cores per task
#SBATCH --mem=16GB                     # Memory per node
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=ALL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=huai.khor@oulu.fi  # Where to send mail
#SBATCH --open-mode=append

source /users/khorhuai/.bashrc

# Load necessary modules
module load LUMI/23.09 partition/L
module load PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240315


# turn off sdpa attention for huggingface models since it doesnt compatible with lumi
export XFORMERS_DISABLED=1

# storage and filenames
# Get the current date and time
current_date_time=$(date +"%Y%m%d_%H%M%S")
# store output and error inside a subfolder
subfolder_for_slurm = "slurm_status/"
# Create a SLURM script with the required output and error file paths
slurm_script="slurm_job_${current_date_time}.sh"

# paths
#sif_path="/projappl/project_462000442/ice_envs/rocm_pytorch.sif"
cache_path="/projappl/project_462000771/ice_envs/cache_dir"

export HIP_VISIBLE_DEVICES=0

#export NCCL_DEBUG=INFO

# rocm check
rocm-smi --showmeminfo vram
#srun singularity exec $SIF conda-python-simple accelerate config
#srun singularity exec $SIF conda-python-simple -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 train_bp4d.py 
srun singularity exec $SIF conda-python-simple train.py \
-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2

# enter the container
#singularity shell $SIF

# activate conda
#$WITH_CONDA
