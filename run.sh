#!/bin/bash
#SBATCH --account=project_2014146
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:1,nvme:10

module purge
module load pytorch

set -x

tar xf /scratch/project_2014146/janne-kauppila/data.tar -C $LOCAL_SCRATCH

srun python3 train2.py --data_path=$LOCAL_SCRATCH/project_2014146/janne-kauppila/data/train --output_dir=$LOCAL_SCRATCH/project_2014146/janne-kauppila/output