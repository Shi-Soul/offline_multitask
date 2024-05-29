#!/bin/bash
set -e
# Each GPU can run 3 job at a time
# run in background, not blocking

# example command
# XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python train_il.py train --random_noise=-1 --ADD_TASK_BIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True

# sweep over all the tasks

echo "BC: Exp on Random Noise"
echo "-1, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1"

echo "Part1"
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=-1 --ADD_TASK_BIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True 
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.00001  --ADD_TASK_BIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.0001  --ADD_TASK_BIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.001  --ADD_TASK_BIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.01  --ADD_TASK_BIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.1  --ADD_TASK_BIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True

 
echo "Part2"
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=-1 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m" --TEST_AFTER_TRAINING=True
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.00001 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m" --TEST_AFTER_TRAINING=True
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.0001 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m" --TEST_AFTER_TRAINING=True
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.001 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m" --TEST_AFTER_TRAINING=True
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.01 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m" --TEST_AFTER_TRAINING=True
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.1 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m" --TEST_AFTER_TRAINING=True
#

echo "Part3"
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=-1 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m,walk_m" --TEST_AFTER_TRAINING=True
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.00001 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m,walk_m" --TEST_AFTER_TRAINING=True
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.0001 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m,walk_m" --TEST_AFTER_TRAINING=True
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.001 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m,walk_m" --TEST_AFTER_TRAINING=True
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.01 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m,walk_m" --TEST_AFTER_TRAINING=True
XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=1 python train_il.py train --random_noise=0.1 --ADD_TASK_BIT=True --USE_DATASET_STR="run_m,walk_m" --TEST_AFTER_TRAINING=True



wait