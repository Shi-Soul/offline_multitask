from typing import Any, Callable, Union
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
import functools

SEED=1
PWD = os.path.dirname(os.path.abspath(__file__))
device = jax.devices("gpu")[0]
assert device.platform=="gpu"


class CQLSACAgent:
    def __init__(self, modelstate, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.modelstate = modelstate
        self.task_bit = 1
        
    def set_task_bit(self, task_bit):
        self.task_bit = task_bit
        
    def act(self, state):
        assert state.shape == (self.state_dim,)
        state = np.concatenate([state, np.ones((1))*self.task_bit], axis=0).reshape(1, -1)
        # action = np.random.uniform(-5, 5, size=(self.action_dim))
        state = jax.device_put(state, device)
        action = self.modelstate.apply_fn({'params': self.modelstate.params}, state)
        return np.array(action[0])
    
    def load(self, load_path):
        pass


def reparameterize(mu, log_std, rng):
    std = jnp.exp(log_std)
    return mu + std * random.normal(rng, mu.shape)

def log_pi(mu, log_std, act):
    std = jnp.exp(log_std)
    return jnp.sum(-0.5 * jnp.log(2 * jnp.pi) - log_std - 0.5 * jnp.square((act - mu) / std), axis=-1)


class Actor(nn.Module):
    act_dims: int
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.silu(nn.Dense(256)(x))
        x = nn.LayerNorm()(x)
        x = nn.silu(nn.Dense(256)(x))+x
        x = nn.Dense(self.act_dims*2)(x)
        # mu and log_std
        return x
    
class Critic(nn.Module):
    @nn.compact
    def __call__(self, x, a):
        x = x.reshape((x.shape[0], -1))
        a = a.reshape((a.shape[0], -1))
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.silu(nn.Dense(256)(x))
        x = nn.LayerNorm()(x)
        x = nn.silu(nn.Dense(256)(x))+x
        x = nn.Dense(1)(x)
        return x
    
class SAC(nn.Module):
    act_dims: int
    
    def setup(self):
        self.actor = Actor(act_dims=self.act_dims)
        self.critic1 = Critic()
        # self.critic2 = Critic()
        
    def __call__(self, x, rng):
        mu,log_std = jnp.split(self.actor(x).reshape(-1, self.act_dims, 2), 2, axis=-1)
        act = reparameterize(mu, log_std, rng)
        q = self.critic1(x, act)
        return q, mu, log_std




@struct.dataclass
class Metrics(metrics.Collection):
    # accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')    
    
class TrainState(train_state.TrainState):
    actor_fn: Callable = struct.field(pytree_node=False)
    critic1_fn: Callable = struct.field(pytree_node=False)
    metrics: Metrics
    
    act_dims: int
    obs_dims: int
    gamma: float
    min_q_weight: float
    num_act_samples: int
    
def create_train_state(rng:jax.Array, learning_rate, momentum, gamma=0.99, min_q_weight=0.5, num_act_samples=10, obs_dims=OBS_DIM, act_dims=ACT_DIM):
    """Creates an initial `TrainState`."""
    model = SAC(ACT_DIM)
    print(model.tabulate(rng, jnp.ones([1, OBS_DIM+1]),rng,compute_flops=True))
    params = model.init(rng, jnp.ones([1, OBS_DIM+1]),rng)['params'] # initialize parameters by passing a template image
    tx = optax.adam(learning_rate, momentum)
    # tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=model.apply, 
        params=params, tx=tx, 
        actor_fn = Actor(model.act_dims).apply,
        critic1_fn = Critic().apply,
        metrics=Metrics.empty(),
        act_dims=act_dims,
        obs_dims=obs_dims,
        gamma=gamma,
        min_q_weight=min_q_weight,
        num_act_samples=num_act_samples
        )

@jax.jit
def train_step(state:TrainState, batch,rng):
    """Train for a single step."""
    
    def actor_loss_fn(actor_params,x, rng):
        q, mu, log_std = state.actor_fn({"params":actor_params},x, rng)
        log_pi = log_pi(mu, log_std, mu)
        return (-q + log_pi).mean()

    def critic_loss_fn(actor_params,critic_params,s, a, r, s_prime, a_prime, rng):
        rng1, rng2 = random.split(rng)
        critic1 = functools.partial(state.critic1_fn, {"params":critic_params})
        actor = functools.partial(state.actor_fn, {"params":actor_params})
        
        # q1 = state.critic1_fn({"params":critic_params},s, a)
        q1 = critic1(s, a)
        q_prime = critic1(s_prime, a_prime)
        td_target = r + state.gamma * q_prime
        td_loss = optax.l2_loss(q1, td_target).mean()
        
        mu,logstd = state.actor(s)
        # sample action from current policy and uniform random
        replicate_mu = jnp.repeat(mu, state.num_act_samples, axis=0)
        replicate_logstd = jnp.repeat(logstd, state.num_act_samples, axis=0)
        # shape: (batch_size*num_act_samples, act_dim)
        policy_act = reparameterize(replicate_mu, replicate_logstd, rng1)
        policy_random = random.uniform(rng2, shape=policy_act.shape, minval=-1, maxval=1)
        
        q_policy = critic1(s, policy_act)
        q_random = critic1(s, policy_random)
        min_q_loss = (q_policy - q_random).mean()
        
        return td_loss + min_q_loss*state.min_q_weight

    def compute_loss(params,s, a, r, s_prime, a_prime, rng):
        rng1, rng2 = random.split(rng)
        actor_loss = actor_loss_fn(params['actor'],s, rng1)
        critic_loss = critic_loss_fn(params,s, a, r, s_prime, a_prime, rng2)
        return actor_loss + critic_loss
    
    def loss_fn(params):
        # pred = state.apply_fn({'params': params}, batch['obs'])
        # loss = optax.l2_loss(pred, batch['act']).mean()
        # loss = state.compute_loss({'params':params},batch['obs'], batch['act'], batch['rew'], batch['obs_prime'], batch['act_prime'], rng)
        loss = compute_loss(params,batch['obs'], batch['act'], batch['rew'], batch['obs_prime'], batch['act_prime'], rng)
        return loss
    
    
    # grad_fn = jax.grad(loss_fn)
    # grads = grad_fn(state.params)
    loss,grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    # state.metrics = state.metrics.merge(Metrics.from_model_output(loss=loss))
    state = state.replace(metrics=state.metrics.merge(Metrics.from_model_output(loss=loss)))
    return state

def train(SAMPLE_EXAMPLE=False,):
    np.random.seed(SEED)
    init_rng = jax.random.key(SEED)
    init_rng, rng1, rng2, rng3 = random.split(init_rng, 4)

    num_epochs = 100
    learning_rate = 0.002
    momentum = 0.9
    batch_size = 1024
    test_size = 0.2
    
    data = make_dataset(MAKE_SARSA=True)
    merge_data = merge_dataset(data['walk_mr'])
    
    train_ds = DataLoader(merge_data, batch_size=batch_size)
    
    learning_rate = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=num_epochs*len(train_ds), alpha=0.0)
    
    state = create_train_state(rng2, 
                               learning_rate, momentum, 
                               gamma=0.99, min_q_weight=0.5, num_act_samples=10)
    state = jax.device_put(state, device)
    
    metrics_history = {'train_loss': []}
    
    for epoch in range(num_epochs):
        
        for step,batch in enumerate(train_ds):
            rng1, rng2 = random.split(rng1, 2)
            state = train_step(state, batch, rng2)
            
        for metric,value in state.metrics.compute().items(): # compute metrics
            metrics_history[f'train_{metric}'].append(value) # record metrics
        
        state = state.replace(metrics=state.metrics.empty()) 
        
        print(f"train epoch\t: {epoch}, "
            f"\tloss: {metrics_history['train_loss'][-1]}, ")
        
    checkpoints.save_checkpoint(ckpt_dir=os.path.join(PWD, 'ckpt','cql'),
                            target=state,
                            step=0,
                            overwrite=True,
                            keep=2)
    
    if SAMPLE_EXAMPLE:
        # sample a batch of data
        batch = next(train_ds)
        # inference
        num_sample = 10 
        act = state.apply_fn({'params': state.params}, batch['obs'][:num_sample])
        loss = optax.l2_loss(act, batch['act'][:num_sample]).mean()
        print("DEBUG: batch['obs']", batch['obs'][:num_sample])
        print("DEBUG: act", act)
        print("DEBUG: batch['act']", batch['act'][:num_sample])
        print("DEBUG: difference", loss)
    
            
    ...
    
def test():
    raise NotImplementedError
    ...

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