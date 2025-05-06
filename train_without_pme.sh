#!/bin/bash
#SBATCH --job-name=meformer-train
#SBATCH --output=runs/without_pme_%j.out
#SBATCH --error=runs/without_pme_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kmbhatt@wpi.edu

#SBATCH -p short             # academic partition
#SBATCH -C A100              # request A100 GPUs (adjust if needed)
#SBATCH --gres=gpu:4         # request 4 GPUs
#SBATCH --ntasks=4           # 1 task per GPU
#SBATCH --cpus-per-task=8    # number of CPU threads per task
#SBATCH --mem=64G            # total memory
#SBATCH -t 1-23:59:00        # 2 days wall time

# Load your modules
module load cudnn8.1-cuda11.2/8.1.1.33

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MEFormer

# Go to repo
cd ~/home/dr_sem4/gits/MEFormer
export PYTHONPATH=`pwd`:$PYTHONPATH

wandb login 

# Launch training using distributed launcher
bash tools/dist_train.sh \
  projects/configs/moad_voxel0075_vov_1600x640_cbgs.py 4
