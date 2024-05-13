import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax.training import checkpoints
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers
import numpy as np
from functools import partial
import fire
import time
from util import *
import os

SEED=1
PWD = os.path.dirname(os.path.abspath(__file__))
device = jax.devices("gpu")[0]
assert device.platform=="gpu"

# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VAE model definitions."""

from flax import linen as nn
from jax import random
import jax.numpy as jnp


# Model: stochastic model, (state,action)+(random noise)->(nextstate,reward)
class Encoder(nn.Module):
  """VAE Encoder."""

  latents: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(256, name='fc1')(x)
    x = nn.relu(x)
    mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
    logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
    return mean_x, logvar_x


class Decoder(nn.Module):
  """VAE Decoder."""
  out_dims: int = OBS_DIM+2

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(256, name='fc1')(z)
    z = nn.relu(z)
    z = nn.Dense(self.out_dims, name='fc2')(z)
    return z


class VAE(nn.Module):
  """Full VAE model."""

  latents: int = 20

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder()

  def __call__(self, s,a, z_rng):
    x = jnp.concatenate([s,a], axis=-1)
    mean, logvar = self.encoder(x)
    z = reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar

  def generate(self, z):
    return self.decoder(z)

total_loss_fn = lambda rec_loss,kld_loss: rec_loss + kld_loss*0.01

def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std

@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@struct.dataclass
class Metrics(metrics.Collection):
    loss_mean: metrics.Average.from_output('loss_mean')
    loss_max : metrics.Average.from_output('loss_max')
    kld_loss : metrics.Average.from_output('kld_loss')
    rec_loss : metrics.Average.from_output('rec_loss')
    
class TrainState(train_state.TrainState):
    metrics: Metrics

def create_train_state(module, rng, learning_rate):
    """Creates an initial `TrainState`."""
    rng1, rng2 = random.split(rng)
    params = module.init(rng1, jnp.ones([1, OBS_DIM+1]),jnp.ones([1,ACT_DIM]),rng2)['params'] # initialize parameters by passing a template image
    tx = optax.adam(learning_rate)
    # tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx,
        metrics=Metrics.empty())

@jax.jit
def compute_metrics(*, state, batch, rng):
    # logits = state.apply_fn({'params': state.params}, batch['obs'])
    # loss = optax.l2_loss(logits, batch['act']).mean()
    pred, mean,logvar = state.apply_fn({'params': state.params}, batch['obs'], batch['act'], rng)
    kld_loss = kl_divergence(mean, logvar).mean()
    rec_loss = optax.l2_loss(pred, jnp.concatenate([batch['obs_prime'],batch['rew']],axis=-1))#.mean()
    # loss = optax.huber_loss(pred, jnp.concatenate([batch['obs_prime'],batch['rew']],axis=-1))#.mean()
    loss = total_loss_fn(rec_loss,kld_loss)
    loss_mean = loss.mean()
    loss_max = loss.max()
    metric_updates = state.metrics.single_from_model_output(
        loss_mean=loss_mean,loss_max=loss_max,rec_loss=rec_loss,kld_loss=kld_loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    # print(loss)
    return state

@jax.jit
def train_step(state, batch, rng):
    """Train for a single step."""
    def loss_fn(params):
        pred, mean,logvar = state.apply_fn({'params': params}, batch['obs'], batch['act'], rng)
        kld_loss = kl_divergence(mean, logvar).mean()
        rec_loss = optax.l2_loss(pred, jnp.concatenate([batch['obs_prime'],batch['rew']],axis=-1)).mean()
        loss = total_loss_fn(rec_loss,kld_loss)
        # loss = optax.huber_loss(pred, jnp.concatenate([batch['obs_prime'],batch['rew']],axis=-1)).mean()
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state



def train(SAMPLE_EXAMPLE=False,TRAIN_TEST_SPLIT=True,VERBOSE=1):

    np.random.seed(SEED)
    init_rng = jax.random.key(SEED)
    init_rng, rng1, rng2, rng3 = random.split(init_rng, 4)

    latents = 6
    # num_epochs = 500
    num_epochs = 30
    learning_rate = 0.002
    batch_size = 256
    test_size = 0.05
    random_noise = -1
    
    
    data = make_dataset(True,True,"remove")
    # merge_data = merge_dataset(data['run_m'],data['run_mr'])
    merge_data = merge_dataset(*data.values())
    # merge_data = merge_dataset(data['walk_mr'],data['walk_m'])
    if TRAIN_TEST_SPLIT:
        # split data into train and test
        train_data, test_data = train_test_split(merge_data, test_size=test_size)
        train_ds = DataLoader(train_data, batch_size=batch_size, random_noise=random_noise)
        test_ds = DataLoader(test_data, batch_size=batch_size, random_noise=-1)
    else:
        train_ds = DataLoader(merge_data, batch_size=batch_size, random_noise=random_noise)
    
    
    # learning_rate = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=num_epochs*len(train_ds), alpha=0.0)
    model = VAE(latents)
    print(model.tabulate(init_rng, 
                         jnp.ones([1, OBS_DIM+1]),jnp.ones([1,ACT_DIM]),rng3,
                         compute_flops=True))
    state = create_train_state(model, rng2, learning_rate)
    state = jax.device_put(state, device)
    del init_rng  # Must not be used anymore.
    

    # metrics_history = {'train_loss': []}

    for epoch in range(num_epochs):
        for step,batch in enumerate(train_ds):
            rng1,key1,key2 = random.split(rng1,3)
            # Run optimization steps over training batches and compute batch metrics
            state = train_step(state, batch,key1) # get updated train state (which contains the updated parameters)
            state = compute_metrics(state=state, batch=batch, rng=key2) # aggregate batch metrics

        # for metric,value in state.metrics.compute().items(): # compute metrics
        #     metrics_history[f'train_{metric}'].append(value) # record metrics
        # state = state.replace(metrics=Metrics.empty())
        metric_res = state.metrics.compute()
        state = state.replace(metrics=state.metrics.empty()) 

        if VERBOSE>=1:
            print(f"train epoch\t: {epoch}, \n",
                  *[f"\t{key:<20}\t: {value}, \n" for key,value in metric_res.items()]
                  )
        
        if TRAIN_TEST_SPLIT:
            for step,batch in enumerate(test_ds):
                rng1,key1 = random.split(rng1)
                state = compute_metrics(state=state, batch=batch, rng=key1)
            metric_res = state.metrics.compute()
            print(f"test  epoch\t: {epoch}, \n",
                  *[f"\t{key:<20}\t: {value}, \n" for key,value in metric_res.items()]
                  )
            state = state.replace(metrics=state.metrics.empty()) 
        
    # Save the model
    checkpoints.save_checkpoint(ckpt_dir=os.path.join(PWD, 'ckpt','mbvae'),
                            target=state,
                            step=0,
                            overwrite=True,
                            keep=2)
    
    if SAMPLE_EXAMPLE:
        batch = next(train_ds)
        rng1,key1 = random.split(rng1)
        num_sample =3 
        pred, mean,logvar = state.apply_fn({'params': state.params}, 
                                           batch['obs'][:num_sample], batch['act'][:num_sample], key1)
        kld_loss = kl_divergence(mean, logvar).mean()
        # rec_loss = optax.l2_loss(pred, jnp.concatenate([batch['obs_prime'],batch['rew']],axis=-1))#.mean()
        # loss = optax.huber_loss(pred, jnp.concatenate([batch['obs_prime'],batch['rew']],axis=-1))#.mean()
        target = jnp.concatenate([batch['obs_prime'][:num_sample],batch['rew'][:num_sample]],axis=-1)
        rec_loss = optax.l2_loss(pred, target).mean()
        loss = total_loss_fn(rec_loss,kld_loss)
        print("DEBUG: batch['obs']", batch['obs'][:num_sample])
        print("DEBUG: batch['act']", batch['act'][:num_sample])
        print("DEBUG: pred", pred)
        print("DEBUG: target", target)
        print("DEBUG: loss", loss)
        print("DEBUG: kld_loss", kld_loss)
        print("DEBUG: rec_loss", rec_loss)
        

def test():
    raise NotImplementedError
    np.random.seed(SEED)
    init_rng = jax.random.key(SEED)

    learning_rate = 0.005
    momentum = 0.9
    
    model = MLP(ACT_DIM)
    state = create_train_state(model, init_rng, learning_rate, momentum)
    state = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(PWD, 'ckpt','mbvae'), target=state)
    state = jax.device_put(state, device)
    # inference
    # act = state.apply_fn({'params': state.params}, batch['obs'])
    agent = MLPAgent(state, OBS_DIM, ACT_DIM)
    eval_agent(agent, eval_episodes=5,seed=SEED)
    
        
if __name__=="__main__":
    try:
        start_time = time.time()
        fire.Fire({
            'train': train,
            'test': test
        })
        end_time = time.time()
        print("Program Running time: ", end_time-start_time)
    except Exception as e:
        print(">>BUG: ",e)
        import pdb;pdb.post_mortem()
