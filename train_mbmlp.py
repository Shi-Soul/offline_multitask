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

# Model: deterministic model, (state,action)->(nextstate,reward)
class MLP(nn.Module):                    # create a Flax Module dataclass
    out_dims: int

    @nn.compact
    def __call__(self, s,a):
        x = jnp.concatenate([s,a], axis=-1)
        x = x.reshape((x.shape[0], -1))
        # x = nn.silu(nn.Dense(64)(x))        # create inline Flax Module submodules
        # x = nn.Dropout(0.1, deterministic=False)(x)
        # x = nn.relu(nn.Dense(32)(x))
        # x = nn.silu(nn.Dense(8192)(x))
        # x = nn.LayerNorm()(x)
        x = nn.silu(nn.Dense(256)(x))
        x = nn.LayerNorm()(x)
        z = nn.silu(nn.Dense(12)(x))
        x = nn.silu(nn.Dense(256)(z))+x
        x = nn.LayerNorm()(x)
        # x = nn.silu(nn.Dense(256)(x))+x
        # x = nn.LayerNorm()(x)
        # x = nn.silu(nn.Dense(256)(x))+x
        # x = nn.LayerNorm()(x)
        # x = nn.silu(nn.Dense(256)(x))+x
        # x = nn.LayerNorm()(x)
        x = nn.Dense(self.out_dims)(x)       # shape inference
        # x = nn.tanh(x)
        return x

@struct.dataclass
class Metrics(metrics.Collection):
    loss_mean: metrics.Average.from_output('loss_mean')
    loss_max : metrics.Average.from_output('loss_max')
    
class TrainState(train_state.TrainState):
    metrics: Metrics

def create_train_state(module, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([1, OBS_DIM+1]),jnp.ones([1,ACT_DIM]))['params'] # initialize parameters by passing a template image
    tx = optax.adam(learning_rate, momentum)
    # tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx,
        metrics=Metrics.empty())

@jax.jit
def compute_metrics(*, state, batch, weight=1.0):
    # logits = state.apply_fn({'params': state.params}, batch['obs'])
    # loss = optax.l2_loss(logits, batch['act']).mean()
    pred = state.apply_fn({'params': state.params}, batch['obs'], batch['act'])
    # loss = optax.l2_loss(pred, jnp.concatenate([batch['obs_prime'],batch['rew']],axis=-1))#.mean()
    loss = weight*optax.huber_loss(pred, jnp.concatenate([batch['obs_prime'],batch['rew']],axis=-1))#.mean()
    loss_mean = loss.mean()
    loss_max = loss.max()
    metric_updates = state.metrics.single_from_model_output(
        loss_mean=loss_mean,loss_max=loss_max)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    # print(loss)
    return state

@jax.jit
def train_step(state, batch, weight=1.0):
    """Train for a single step."""
    def loss_fn(params):
        pred = state.apply_fn({'params': params}, batch['obs'], batch['act'])
        # loss = optax.l2_loss(pred, jnp.concatenate([batch['obs_prime'],batch['rew']],axis=-1)).mean()
        
        loss = optax.huber_loss(pred, jnp.concatenate([batch['obs_prime'],batch['rew']],axis=-1)).mean()
        return loss*weight
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state



def train(USE_NEG_BATCH = True,SAMPLE_EXAMPLE=False,TRAIN_TEST_SPLIT=True,VERBOSE=1):

    np.random.seed(SEED)
    init_rng = jax.random.key(SEED)
    
    

    num_epochs = 100
    # num_epochs = 100
    learning_rate = 0.002
    momentum = 0.95
    batch_size = 256
    test_size = 0.05
    random_noise = 0.0003
    
    
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
    
    
    learning_rate = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=num_epochs*len(train_ds), alpha=0.0)
    model = MLP(OBS_DIM+2)
    print(model.tabulate(init_rng, jnp.ones([1, OBS_DIM+1]),jnp.ones([1,ACT_DIM]),compute_flops=True))
    state = create_train_state(model, init_rng, learning_rate, momentum)
    state = jax.device_put(state, device)
    del init_rng  # Must not be used anymore.
    

    # metrics_history = {'train_loss': []}

    for epoch in range(num_epochs):
        for step,batch in enumerate(train_ds):
            # Run optimization steps over training batches and compute batch metrics
            # state = train_step(state, batch) # get updated train state (which contains the updated parameters)
            # state = compute_metrics(state=state, batch=batch) # aggregate batch metrics
            
            if not USE_NEG_BATCH:
            
                state = train_step(state, batch) # get updated train state (which contains the updated parameters)
                state = compute_metrics(state=state, batch=batch) # aggregate batch metrics
            
            else:
                neg_batch = {
                    "obs": np.concatenate([batch["obs"],   batch["obs"]*np.random.randn(*batch["obs"].shape)   ],axis=0),
                    "act": np.concatenate([batch["act"],np.random.uniform(-1,1,batch["act"].shape)],axis=0),
                    "obs_prime": np.concatenate([batch["obs_prime"], np.zeros_like(batch["obs"])],axis=0),
                    "rew":   np.concatenate([batch["rew"], np.zeros_like(batch["rew"])-1 ],axis=0 )
                }
                state = train_step(state, neg_batch, weight = 1.0) 
                state = compute_metrics(state=state, batch=neg_batch, weight = 1.0) 
                # neg_batch = {key: -value for key,value in batch.items()}
                # state = compute_metrics(state=state, batch=neg_batch)

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
                state = compute_metrics(state=state, batch=batch)
            metric_res = state.metrics.compute()
            print(f"test  epoch\t: {epoch}, \n",
                  *[f"\t{key:<20}\t: {value}, \n" for key,value in metric_res.items()]
                  )
            state = state.replace(metrics=state.metrics.empty()) 
        
    # Save the model
    checkpoints.save_checkpoint(ckpt_dir=os.path.join(PWD, 'ckpt','mb'),
                            target=state,
                            step=0,
                            overwrite=True,
                            keep=2)
    
    if SAMPLE_EXAMPLE:
        # sample a batch of data
        batch = next(train_ds)
        # inference
        num_sample =3 
        pred = state.apply_fn({'params': state.params}, batch['obs'][:num_sample], batch['act'][:num_sample])
        target = jnp.concatenate([batch['obs_prime'][:num_sample],batch['rew'][:num_sample]],axis=-1)
        loss = optax.l2_loss(pred, target).mean()
        print("DEBUG: batch['obs']", batch['obs'][:num_sample])
        print("DEBUG: batch['act']", batch['act'][:num_sample])
        print("DEBUG: pred", pred)
        print("DEBUG: target", target)
        print("DEBUG: loss", loss)
        

def test():
    raise NotImplementedError
    np.random.seed(SEED)
    init_rng = jax.random.key(SEED)

    learning_rate = 0.005
    momentum = 0.9
    
    model = MLP(ACT_DIM)
    state = create_train_state(model, init_rng, learning_rate, momentum)
    state = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(PWD, 'ckpt','mb'), target=state)
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
