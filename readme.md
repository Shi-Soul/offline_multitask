
## Overview & Planning

Baseline
- Dataset Expert Performance: 
    - run (318.36557 251.48225), walk (962.8321 929.83185)   
- Random Policy Performance: 
    - walk 56, run 29


- Exp
    - 每个setting 训3次, eval时取100个episodes, 需要保留最佳模型
    - baseline: bc, random 
    - 最佳方法: cql, td3+bc, dt, mb  
    - 有task bit, 无task bit: 
        - walk-> walk; run-> run; **walk+run->walk+run**
    - add noise exp
        - different noise magnitude
        - 
    - Dataset Exp
        - Walk-> Walk; Run-> Run; Walk+Run -> Walk+Run
        - Walk m-> walk, walk mr -> walk;
        - run  m-> run , run  mr -> run;  
        - // (walk m+ run m)





| Method | walk(best) | run(best) |
| -------- | -------- | -------- |
| Expert Traj | 962.8321 | 318.36557 |
| Random | 51.85748167535231 +- 12.673925499073189 | 27.921023864970184 +- 3.4484724900639914 |
| BC   | (181.86798185904092, 44.01772755239101)  | (66.94229235058019, 14.862189258857475) |
| ~~CQL(ours)~~   | 102   | 49   |
| CQL(tianshou)   | 205.78   | 76.17   |
| Dicision Tranformer | 229 | 75 |
| TD3+BC(tianshou)   | 160   | ?   |
| model base ppo | ? | ? |
| morel | ? | ? |
| ~~GAIL(tianshou)~~   | 160   | ?   |
| ~~PPO(sb3,online)~~ | ? | ? |

tianshou_cql results

| dataset | walk(best) | run(best) | step per epoch | add task bit |
| -------- | -------- | -------- | -------- | -------- |
| all | 205.78 | 76.17 | 5000 | true |
| all | 199.10 | 83.52 | 1000 | true |
| all | 211.43 | 87.13 | 1000 | false |

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
- (working) comparison experiments
    - find best parameter for existing algorithms 
    - does more data help?
    - does adding noise help?
- (working) interface
    - 需要把最佳模型包装起来, 做成agent_example能调用的形式


## Experience Results

### Whether to add task bit

| Method | Taskbit | walk | run | walk(all) | run(all) |
| -------- | -------- | -------- | --------  | --------  | --------    |
| random | \\ | ? | ? | ? | ? |
| bc | Yes | ? | ? | ? | ?
| bc | No | ? | ? | ? | ?
| cql | Yes  | ? | ? | ? | ? |
| cql | No | ? | ? | ? | ? |




### Whether to add noise

| Method | noise | walk | run | walk(all) | run(all) |
| -------- | -------- | -------- | --------  | --------  | --------    |
| cql | 0  | ? | ? | ? | ? |
| cql | 0.1 | ? | ? | ? | ? |

### Different dataset

| Method | walk | run | walk(all) | run(all) | walk m | walk mr | run m | run mr|
| -------- | -------- | -------- | --------  | --------  | --------    |-------- | --------  | --------  |
|  bc | ? | ? | ? | ? | ? | ? | ? |
| cql | ? | ? | ? | ? | ? | ? | ? |

<!-- | random | ? | ? | ? | ? | ? | ? | ? | -->

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
```

```bash
# Get Project.zip in this dir
unzip Project.zip project/collected_data/*
mv project/collected_data/ .
rm -r project
```

```bash
# Run random policy and two algorithms
python agent.py
python train_il.py train
python train_il.py test
python train_il.py train --random_noise=-1 --ADD_TASK_BIT=True --USE_DATASET_STR="walk_m" --TEST_AFTER_TRAINING=True

python train_cqlsac.py train
python train_cqlsac.py test

CUDA_VISIBLE_DEVICES=0 python train_morel.py --comet_api wye3SgI6S0uSJyf5Mc54R0DTr --exp_name v2_t3.2_n32
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

