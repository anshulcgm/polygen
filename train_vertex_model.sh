#!/bin/bash

#SBATCH --job-name=train_vertex_model_image_conditional
#SBATCH --output=logs_vertex_model_image_conditional.out
#SBATCH --error=logs_vertex_model_image_conditional.err
#SBATCH --gres gpu:4 --constraint=a40
#SBATCH --partition=overcap
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --account=overcap

srun python -u polygen/training/train_vertex_model.py