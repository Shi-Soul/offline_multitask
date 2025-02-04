#!/bin/bash
set -e
# Each GPU can run 3 job at a time
# run in background, not blocking

# example command
# CUDA_VISIBLE_DEVICES=0 python train_il.py train --random_noise=-1 --ADD_TASK_BIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True

# sweep over all the tasks

# USE/NONUSE taskbit
echo "BC: Exp on USE/NONUSE taskbit"
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python train_il.py train --random_noise=-1 --ADD_TASK_BIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True &
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python train_il.py train --random_noise=-1 --ADD_TASK_BIT=False --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True &

wait 

XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python train_il.py train --random_noise=-1 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m" --TEST_AFTER_TRAINING=True &
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python train_il.py train --random_noise=-1 --ADD_TASK_BIT=False --USE_DATASET_STR="run_m" --TEST_AFTER_TRAINING=True &

wait

XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python train_il.py train --random_noise=-1 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m,walk_m" --TEST_AFTER_TRAINING=True &
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python train_il.py train --random_noise=-1 --ADD_TASK_BIT=False --USE_DATASET_STR="run_m,walk_m" --TEST_AFTER_TRAINING=True &

# Wait for all the jobs to finish
wait