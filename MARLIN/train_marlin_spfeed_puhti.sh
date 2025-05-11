#!/bin/bash
#SBATCH --job-name=job_name         # Job name
#SBATCH --account=project_2005312
#SBATCH --output=%x_%j.out       # Output file
#SBATCH --error=%x_%j.err        # Error file
#SBATCH --time=0:10:00                # Time limit (hh:mm:ss)
#SBATCH --partition=gpusmall           # Partition name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=1GB                     # Memory per node
#SBATCH --mail-type=BEGIN,END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=huai.khor@oulu.fi  # Where to send mail


# module load matlab
module load tykyy

# storage and filenames
# Get the current date and time
current_date_time=$(date +"%Y%m%d_%H%M%S")
# store output and error inside a subfolder
subfolder_for_slurm = "slurm_status/"
# Create a SLURM script with the required output and error file paths
slurm_script="slurm_job_${current_date_time}.sh"

source /users/khorhuai/.bashrc
#conda activate videomamba_py311

# python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Set library path
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip install matlabengine

# Run your application
# python bp4d_preprocessing.py
python train.py

