#!/bin/bash

#SBATCH --job-name=train_face_model_class_conditional
#SBATCH --output=logs_face_model_class_conditional.out
#SBATCH --error=logs_face_model_class_conditional.err
#SBATCH --gres gpu:4 --constraint=a40
#SBATCH --partition=overcap
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --account=overcap

srun python -u polygen/training/train_face_model_class_conditional.py