#!/bin/bash
#SBATCH --job-name=raw-proposals
#SBATCH --output=runs/raw_proposals_%j.out
#SBATCH --error=runs/raw_proposals_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kmbhatt@wpi.edu

#SBATCH -p long        # use the academic partition
#SBATCH -C A100             # request A30 GPUs
#SBATCH --gres=gpu:1       # 1 GPU
#SBATCH --ntasks=1         # single MPI task
#SBATCH --cpus-per-task=8  # threads per task
#SBATCH --mem=32G          # RAM
#SBATCH -t 2:59:00        # walltime ≈ 24 h

# (Optional) load any modules your env needs:
module load cudnn8.1-cuda11.2/8.1.1.33

# Activate your conda environment:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MEFormer

# Go to your repo and ensure your plugin code is on PYTHONPATH
cd ~/home/dr_sem4/gits/MEFormer
export PYTHONPATH=`pwd`:$PYTHONPATH

# Launch the raw‐proposal dump
python tools/test.py \
  projects/configs/meformer_voxel0075_vov_1600x640_cbgs.py \
  ckpts/meformer_voxel0075_vov_1600x640_cbgs.pth \
  --out final_detections.pkl \
