
## Environment Setup

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
pip install stable-baselines3[extra] #2.3.2
pip install imageio
```

```bash
# Get Project.zip in this dir
unzip Project.zip project/collected_data/*
mv project/collected_data/ .
rm -r project
```

## Reference

https://gymnasium.farama.org/environments/mujoco/walker2d/

https://github.com/google-deepmind/dm_control

https://sites.google.com/view/latent-policy

https://flax.readthedocs.io/en/latest/index.html

https://flax.readthedocs.io/en/latest/guides/training_techniques/dropout.html
