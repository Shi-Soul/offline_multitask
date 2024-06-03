
# Offline Multi-task Reinforcement Learning| RL Course Project | SJTU AI3601

## Introduction

In [this repo](https://github.com/Shi-Soul/offline_multitask), we aim to solve the offline multi-task reinforcement learning problem. We use the walker2d environment from dm_control as the testbed, and use collected trajectories for walk and run tasks as the offline dataset. We need to train a RL model to solve both tasks. 

We apply several algorithms, including naive imitation learning, CQL-SAC, TD3+BC, model-based method (naive model based method, MOReL), decision transformer and CDS (implemented in UTDS). We also compare the performance of these algorithms and analyze the results.

We also write some codes for multi-processing parallel evaluation(see `util.py`) and policy visualization. We mainly use [wandb](https://wandb.ai/) to log the training process and results.

We include our best Decision Transformer model as the final agent. The agent can be evaluated in the following way:

```bash 
# Please ensure the collected_data is set up properly, and the model_dt_best.pt is in the root dir.

# To setup dataset, please:
# First Get Project.zip (From project requirement) in this dir
unzip Project.zip project/collected_data/*
mv project/collected_data/ .
rm -r project

cp -r collected_data collected_data_old
python data_transform_.py

# Run evaluation
ls model_dt_best.pt
python decision-transformer/gym/dtagent.py

# Evaluation Output should look like:

# 100%|██████████████████████████████████████████████████████████████████████████████████████| 204/204 [00:00<00:00, 590.41it/s]
# ########### Data Loaded! ###########
# episode_reward [849.8103782379474, 962.2504561873442, 950.071756306005, 939.24570807684, 957.1563025278213, 958.1312475244004, 913.8022988685859, 978.122306614396, 911.985195899648, 951.2113308436805]
# episode_reward_mean 937.1786981086668
# episode_reward_std 35.1068316846696
# episode_length 1000.0
# episode_reward [250.60686935427643, 255.7014766754349, 259.8757880017744, 264.97924390853655, 254.45711360339502, 251.6547112105695, 254.6708868552152, 265.57952390932024, 272.8569586012661, 270.82264443394484]
# episode_reward_mean 260.1205216553733
# episode_reward_std 7.57430020195958
# episode_length 1000.0

```

Code Structure
- BC: `train_il.py`, `train_ilmb.py`
- CQL: `tianshou_cql.py` (tianshou implementation), `train_cqlsac.py` (our implementation)
- TD3BC: `tianshou_td3bc.py`
- Decision Transformer: `decision-transformer/`
- Model based methods: `train_mbmlp.py`, `train_mbvae.py`, `train_morel.py`, `train_ppomb.py`, `morel/`
- UTDS & CDS: `UTDS/`
- Tools code: `util.py`, `data_transform_.py`, `agent.py`, `dmc.py`, `eval.py`, `dmc2gym/`, `exp_scripts/`, ...

Performance of each algorithm:

|	| Walk| 	Run|
|---|----|----|
|Expert|	962.83|	318.37|
| Random|	  51.86 +-   12.67|	  27.92 +-   3.45|
|||
|BC|	854.95 +- 207.23|	**308.08** +- 25.12|
|TD3BC	|933.92 +- 102.27|	275.11 +- 69.71|
|CQL|	934.74 +- 103.25|	291.66 +- 57.67|
|DT|	**956.01** +-   15.34|	258.34 +-   4.34|
|DT w noise|	939.86 +-   26.44|	277.53 +- 22.84|


## Logging & Planning

Baseline
- Dataset Expert Performance: 
    - run (318.36557 251.48225), walk (962.8321 929.83185)   


- (done) Implement  Naive Imitation Learning 
- (done) Implement  CQL-SAC 
    - (problem) which differences make tianshou CQL work?
    - (problem) why log_std_max is around 300? too large
        - when clip log_std into [-20,2], the training is more stable, alpha is higher, other metrics remain unchanged. but performance seems not better
        - try state-independent log_std ? scale*tanh+bias?
    - (future) add more tricks to make it work.
    - (future) try add random noise for better robustness
    - (future) try early stop (at 300 epochs), see if the performance is better
    - (solved) Not so good, critic loss is too large, result in actor's behaviour nearly random
    - (solved) in training, the critic looks like in self-excitation oscillation
        - the game dynamics and expert traj are looped, which is bad for critic to learn
        - r + gamma Q=Q => Q=r/(1-gamma), which cause the Q value to be very large and meaningless
        - td error is very large: in terminal state, target value is super small, but the Q value is very large
        - in offline training, the model have no idea about the terminal state and timestep limit
        - **possible solution**: reward normalization
- (done) Implement Model base method
    - Implemented MLP & VAE model.
    - (problem) how to use the model?
- (done) Implement Decision Transformer
- (done) comparison experiments
    - find best parameter for existing algorithms 
    - does more data help?
    - does adding noise help?
- (done) interface
    - 需要把最佳模型包装起来, 做成agent_example能调用的形式
    -  decision transformer


## Usage

```bash
conda create -n rlp python=3.10 -y
conda activate rlp
pip install dm_control # 1.0.18
pip install gym  # 0.26.2

pip install "jax==0.4.26"
# pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    # For CUDA version Jax
pip install flax #0.8.3
pip install clu #0.0.12
pip install fire #0.6.0
pip install wandb #0.16.6

pip install ray #2.22.0 # For fast eval, useless

# Only for running tianshou and stable-baselines3 for comparison, not necessary
pip install tianshou
pip install stable-baselines3[extra] #2.3.2 
pip install imageio #

# for morel
pip install comet_ml  #3.42.1
pip install "transformers==4.36"
```

```bash
# Get Project.zip in this dir
unzip Project.zip project/collected_data/*
mv project/collected_data/ .
rm -r project
```

```bash
# Run random policy and some algorithms
python agent.py
python train_il.py train --random_noise=-1 --ADD_TASK_BIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True
python tianshou_td3bc.py --random_noise=-1 --ADD_TASKBIT=True --USE_DATASET_STR="walk_m,walk_mr" --task walk
python tianshou_cql.py --random_noise=-1 --ADD_TASKBIT=True --USE_DATASET_STR="__all__" --task walk
python decision-transformer/gym/experiment_dmc.py --USE_DATASET_STR __all__ --task_bit True --noise -1 --save_model True

# python train_cqlsac.py train
# python train_cqlsac.py test

python train_morel.py
```


## Reference

https://wandb.ai/

https://gymnasium.farama.org/environments/mujoco/walker2d/

https://github.com/google-deepmind/dm_control

https://sites.google.com/view/latent-policy

https://flax.readthedocs.io/en/latest/index.html

https://flax.readthedocs.io/en/latest/guides/training_techniques/dropout.html

https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/cql.py

https://github.com/thu-ml/tianshou/

https://github.com/kzl/decision-transformer

https://github.com/Baichenjia/UTDS

https://github.com/denisyarats/dmc2gym

https://github.com/SwapnilPande/MOReL
