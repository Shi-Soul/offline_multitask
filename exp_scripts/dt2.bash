#!/bin/bash
set -e
# Each GPU can run 3 job at a time
# run in background, not blocking

# example command
# XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python tianshou_cql.py --random_noise=-1 --ADD_TASKBIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True

# sweep over all the tasks

echo "CQL: Exp on Random Noise"
echo "-1, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1"

echo "Part1"
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR walk_m,walk_mr --task_bit True --noise -1 --save_model True
# CUDA_VISIBLE_DEVICES=2 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR walk_m,walk_mr --task_bit True --noise 0.00001 --save_model True
# CUDA_VISIBLE_DEVICES=2 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR walk_m,walk_mr --task_bit True --noise 0.0001 --save_model True
# CUDA_VISIBLE_DEVICES=2 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR walk_m,walk_mr --task_bit True --noise 0.001 --save_model True
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR walk_m,walk_mr --task_bit True --noise 0.01 --save_model True
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR walk_m,walk_mr --task_bit True --noise 0.1 --save_model True

 
echo "Part2"
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR walk_m,walk_mr --task_bit True --noise -1 --save_model True
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR run_m,run_mr --task_bit True --noise 0.00001 --save_model True
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR run_m,run_mr --task_bit True --noise 0.0001 --save_model True
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR run_m,run_mr --task_bit True --noise 0.001 --save_model True
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR run_m,run_mr --task_bit True --noise 0.01 --save_model True
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR run_m,run_mr --task_bit True --noise 0.1 --save_model True

echo "Part3"
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR __all__ --task_bit True --noise 0.00001 --save_model True
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR __all__ --task_bit True --noise 0.0001 --save_model True
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR __all__ --task_bit True --noise 0.001 --save_model True
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR __all__ --task_bit True --noise 0.01 --save_model True
# CUDA_VISIBLE_DEVICES=0 python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR __all__ --task_bit True --noise 0.1 --save_model True


wait