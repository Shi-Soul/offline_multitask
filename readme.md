
## Overview & Planning

Baseline
- Dataset Expert Performance: 
    - run (318.36557 251.48225), walk (962.8321 929.83185)   
- Random Policy Performance: 
    - walk 56, run 29

| Method | walk(best) | run(best) |
| -------- | -------- | -------- |
| Expert Traj | 962.8321 | 318.36557 |
| Random | 56 | 29 |
| BC   | 173  | 67   |
| CQL(ours)   | 102   | 49   |
| CQL(tianshou)   | 200   | 200   |
| TD3+BC(tianshou)   | 160   | ?   |
| GAIL(tianshou)   | 160   | ?   |
| ~~PPO(sb3,online)~~ | ? | ? |

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
- (working) Implement Model base method
    - Implemented MLP & VAE model.
    - (problem) how to use the model?
- (working) Implement Decision Transformer
- (future) comparison experiments
    - find best parameter for existing algorithms 
    - does more data help?
    - does adding noise help?



## Usage

```bash
conda create -n rlp python=3.10 -y
conda activate rlp
pip install dm_control # 1.0.18
pip install gym  # 0.26.2

pip install jax # 0.4.26
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
python train_cqlsac.py train
python train_cqlsac.py test

CUDA_VISIBLE_DEVICES=1 python train_morel.py --comet_api wye3SgI6S0uSJyf5Mc54R0DTr 
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

