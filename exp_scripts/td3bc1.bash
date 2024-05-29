#!/bin/bash
set -e
# Each GPU can run 2 job at a time
# run in background, not blocking

# example command
# CUDA_VISIBLE_DEVICES=0 python train_il.py train --random_noise=-1 --ADD_TASKBIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True

# sweep over all the tasks

# USE/NONUSE taskbit
echo "TD3BC: Exp on USE/NONUSE taskbit"
CUDA_VISIBLE_DEVICES=0 python tianshou_td3bc.py --random_noise=-1 --ADD_TASKBIT=True --USE_DATASET_STR="walk_m,walk_mr" --task walk
CUDA_VISIBLE_DEVICES=0 python tianshou_td3bc.py --random_noise=-1 --ADD_TASKBIT=False --USE_DATASET_STR="walk_m,walk_mr" --task walk

CUDA_VISIBLE_DEVICES=0 python tianshou_td3bc.py --random_noise=-1 --ADD_TASKBIT=True --USE_DATASET_STR="run_m,run_mr" --task run
CUDA_VISIBLE_DEVICES=0 python tianshou_td3bc.py  --random_noise=-1 --ADD_TASKBIT=False --USE_DATASET_STR="run_m,run_mr" --task run

CUDA_VISIBLE_DEVICES=0 python tianshou_td3bc.py --random_noise=-1 --ADD_TASKBIT=True --USE_DATASET_STR="__all__" --task walk
CUDA_VISIBLE_DEVICES=0 python tianshou_td3bc.py --random_noise=-1 --ADD_TASKBIT=False --USE_DATASET_STR="__all__" --task walk

# Wait for all the jobs to finish
wait
echo "Finish:"