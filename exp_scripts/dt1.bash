#!/bin/bash
set -e
# Each GPU can run 2 job at a time
# run in background, not blocking

# example command
# CUDA_VISIBLE_DEVICES=0 python train_il.py train --random_noise=-1 --ADD_TASKBIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True
# cd ~/wjxie/env/offline_multitask; conda activate rlp2

# sweep over all the tasks

# USE/NONUSE taskbit
echo "TD3BC: Exp on USE/NONUSE taskbit"
CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR __all__ --task_bit False --noise -1 --save_model True
CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR __all__ --task_bit True --noise -1 --save_model True

CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR walk_m,walk_mr --task_bit False --noise -1 --save_model True
CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR walk_m,walk_mr --task_bit True --noise -1 --save_model True

CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR run_m,run_mr --task_bit False --noise -1 --save_model True
CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR run_m,run_mr --task_bit True --noise -1 --save_model True

# Wait for all the jobs to finish
wait
echo "Finish:"