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
        mu,log_std = jnp.split(x.reshape(-1, self.act_dims, 2),2, axis=-1)
        return mu,log_std

    
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
    obs_dims: int
    act_dims: int
    gamma: float
    min_q_weight: float
    num_act_samples: int
    
    def setup(self):
        self.actor = Actor(act_dims=self.act_dims)
        self.critic1 = Critic()
        # self.critic2 = Critic()
        
    def __call__(self, x, rng):
        # mu,log_std = jnp.split(self.actor(x).reshape(-1, self.act_dims, 2),2, axis=-1)
        mu,log_std = self.actor(x)
        act = self.reparameterize(mu, log_std, rng)
        q = self.critic1(x, act)
        return q, mu, log_std
    
    def reparameterize(self, mu, log_std, rng):
        std = jnp.exp(log_std)
        return mu + std * random.normal(rng, mu.shape)

    def log_pi(self, mu, log_std, act):
        std = jnp.exp(log_std)
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi) - log_std - 0.5 * jnp.square((act - mu) / std), axis=-1)

    def actor_loss(self, x, rng):
        q, mu, log_std = self(x, rng)
        log_pi = self.log_pi(mu, log_std, mu)
        return (-q + log_pi).mean()
    
    def critic_loss(self, s, a, r, s_prime, a_prime, rng):
        rng1, rng2 = random.split(rng)
        
        q1 = self.critic1(s, a)
        q_prime = self.critic1(s_prime, a_prime)
        td_target = r + self.gamma * q_prime
        td_loss = optax.l2_loss(q1, td_target).mean()
        
        mu,logstd = self.actor(s)
        # sample action from current policy and uniform random
        replicate_mu = jnp.repeat(mu, self.num_act_samples, axis=0)
        replicate_logstd = jnp.repeat(logstd, self.num_act_samples, axis=0)
        # shape: (batch_size*num_act_samples, act_dim)
        policy_act = self.reparameterize(replicate_mu, replicate_logstd, rng1)
        policy_random = random.uniform(rng2, shape=policy_act.shape, minval=-1, maxval=1)
        
        replicate_s = jnp.repeat(s, self.num_act_samples, axis=0)
        q_policy = self.critic1(replicate_s, policy_act)
        q_random = self.critic1(replicate_s, policy_random)
        min_q_loss = (q_policy - q_random).mean()
        
        return td_loss + min_q_loss*self.min_q_weight
    
    def compute_loss(self, s, a, r, s_prime, a_prime, rng):
        rng1, rng2 = random.split(rng)
        actor_loss = self.actor_loss(s, rng1)
        critic_loss = self.critic_loss(s, a, r, s_prime, a_prime, rng2)
        return actor_loss + critic_loss

@struct.dataclass
class Metrics(metrics.Collection):
    # accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')    
    
class TrainState(train_state.TrainState):
    
    metrics: Metrics
    
def create_train_state(model:nn.Module ,rng:jax.Array, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = model.init(rng, jnp.ones([1, OBS_DIM+1]),rng)['params'] # initialize parameters by passing a template image
    tx = optax.adam(learning_rate, momentum)
    # tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
        metrics=Metrics.empty())

@jax.jit
def train_step(state, batch,rng):
    """Train for a single step."""
    def loss_fn(params):
        # pred = state.apply_fn({'params': params}, batch['obs'])
        # loss = optax.l2_loss(pred, batch['act']).mean()
        loss = state.apply_fn({'params':params},batch['obs'], batch['act'], batch['rew'], batch['obs_prime'], batch['act_prime'], rng,method='compute_loss')
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
    batch_size = 512
    test_size = 0.2
    
    data = make_dataset(MAKE_SARSA=True)
    merge_data = merge_dataset(data['walk_mr'])
    train_ds = DataLoader(merge_data, batch_size=batch_size)
    
    learning_rate = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=num_epochs*len(train_ds), alpha=0.0)
    
    model = SAC(obs_dims=OBS_DIM, act_dims=ACT_DIM, gamma=0.99, min_q_weight=0.5, num_act_samples=10)
    print(model.tabulate(rng3, jnp.ones([1, OBS_DIM+1]),rng3,compute_flops=True))
    state = create_train_state(model, rng2, learning_rate, momentum)
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