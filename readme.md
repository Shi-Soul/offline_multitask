
## Overview & Planning

Baseline
- Dataset Expert Performance: 
    - run (318.36557 251.48225), walk (962.8321 929.83185)   
- Random Policy Performance: 
    - walk 56, run 29

What have been done:
- Implement the Naive Imitation Learning algorithm
    - Score: 131 (Walk), 53 (Run)
- Implement the CQL-SAC algorithm
    - Not so good, critic loss is too large, result in actor's behaviour nearly random
    - in training, the critic looks like in self-excitation oscillation
        - the game dynamics and expert traj are looped, which is bad for critic to learn
        - r + gamma Q=Q => Q=r/(1-gamma), which cause the Q value to be very large and meaningless
        - td error is very large: in terminal state, target value is super small, but the Q value is very large
        - in offline training, the model have no idea about the terminal state and timestep limit
        - **possible solution**: reward normalization
    - when add tanh to output of actor, nan appear in around 30 epochs

What may be done in future:
- Try to finetune the CQL-SAC algorithm, add some tricks to improve the performance
- Try to implement model base methods
    - First try deterministic env model, train a ppo / sac 
    - If stochastic env model is needed, try to train a vae to predict the next state


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

# Only for running stable-baselines3, not necessary
pip install stable-baselines3[extra] #2.3.2 
pip install imageio #
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
```



## Reference

https://gymnasium.farama.org/environments/mujoco/walker2d/

https://github.com/google-deepmind/dm_control

https://sites.google.com/view/latent-policy

https://flax.readthedocs.io/en/latest/index.html

https://flax.readthedocs.io/en/latest/guides/training_techniques/dropout.html

https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/cql.py
