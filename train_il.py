
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random
import sys
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

default_seed=1
PWD = os.path.dirname(os.path.abspath(__file__))
device = jax.devices("cpu")[0]
# device = jax.devices("gpu")[0]
# assert device.platform=="gpu"
ind = time.strftime("%Y%m%d-%H%M%S")+str(np.random.randint(1000))
CKPT_DIR = os.path.join(PWD, 'ckpt','il',ind)

class MLPAgent:
    # An example of the agent to be implemented.
    # Your agent must extend from this class (you may add any other functions if needed).
    def __init__(self, modelstate, state_dim, action_dim,ADD_TASK_BIT=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.modelstate = modelstate
        self.ADD_TASK_BIT = ADD_TASK_BIT
        self.task_bit = 1
        
    def set_task_bit(self, task_bit):
        print("DEBUG: set_task_bit", task_bit)
        self.task_bit = task_bit
        
    def act(self, state):
        assert state.shape == (self.state_dim,)
        if self.ADD_TASK_BIT:
            state = np.concatenate([state, np.ones((1))*self.task_bit], axis=0).reshape(1, -1)
        else:
            state = state.reshape(1, -1)
        # action = np.random.uniform(-5, 5, size=(self.action_dim))
        state = jax.device_put(state, device)
        action = self.modelstate.apply_fn({'params': self.modelstate.params}, state)
        return np.array(action[0])
    
    def act_vec(self, states):
        assert states.shape[1] == self.state_dim
        if self.ADD_TASK_BIT:
            states = np.concatenate([states, np.ones((states.shape[0],1))*self.task_bit], axis=1)
        states = jax.device_put(states, device)
        actions = self.modelstate.apply_fn({'params': self.modelstate.params}, states)
        return np.array(actions)
    
    def load(self, load_path):
        pass

# Model: deterministic policy, imitation learning, MSE loss
class MLP(nn.Module):                    # create a Flax Module dataclass
    out_dims: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        # x = nn.silu(nn.Dense(64)(x))        # create inline Flax Module submodules
        # x = nn.Dropout(0.1, deterministic=False)(x)
        # x = nn.relu(nn.Dense(32)(x))
        x = nn.silu(nn.Dense(256)(x))
        x = nn.LayerNorm()(x)
        x = nn.silu(nn.Dense(256)(x))+x
        # x = nn.LayerNorm()(x)
        # x = nn.silu(nn.Dense(256)(x))+x
        # x = nn.LayerNorm()(x)
        # x = nn.silu(nn.Dense(64)(x))
        # x = nn.LayerNorm()(x)
        # x = nn.silu(nn.Dense(64)(x))+x
        # x = nn.LayerNorm()(x)
        # x = nn.silu(nn.Dense(64)(x))+x
        # x = nn.LayerNorm()(x)
        x = nn.Dense(self.out_dims)(x)       # shape inference
        x = nn.tanh(x)
        return x

@struct.dataclass
class Metrics(metrics.Collection):
    # accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')
    
class TrainState(train_state.TrainState):
    metrics: Metrics

def create_train_state(module, rng, learning_rate, momentum,ADD_TASK_BIT):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([1, OBS_DIM+int(ADD_TASK_BIT)]))['params'] # initialize parameters by passing a template image
    tx = optax.adam(learning_rate, momentum)
    # tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx,
        metrics=Metrics.empty())

@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({'params': state.params}, batch['obs'])
    loss = optax.l2_loss(logits, batch['act']).mean()
    metric_updates = state.metrics.single_from_model_output(
        loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    # print(loss)
    return state

@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        pred = state.apply_fn({'params': params}, batch['obs'])
        loss = optax.l2_loss(pred, batch['act']).mean()
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


def main(random_noise=-1,num_epochs=100,
         ADD_TASK_BIT=True,
         USE_DATASET_STR="__all__",
         SAMPLE_EXAMPLE=False,
         TRAIN_TEST_SPLIT=False,
         TEST_AFTER_TRAINING=False,
         SEED=default_seed,
         ):

    np.random.seed(SEED)
    init_rng = jax.random.key(SEED)

    num_epochs = num_epochs
    # num_epochs = 100
    learning_rate = 0.002
    momentum = 0.9
    batch_size = 1024
    test_size = 0.2
    random_noise = random_noise
    
    
    # data = make_dataset()
    # # merge_data = merge_dataset(data['run_m'],data['walk_m'])
    # # merge_data = merge_dataset(*data.values())
    # merge_data = merge_dataset(data['walk_mr'])
    if isinstance(USE_DATASET_STR, str):
        USE_DATASET_STR = [USE_DATASET_STR]
    merge_data = make_merge_dataset(USE_DATASET_STR, ADD_TASKBIT=ADD_TASK_BIT)
    if TRAIN_TEST_SPLIT:
        # split data into train and test
        train_data, test_data = train_test_split(merge_data, test_size=test_size)
        train_ds = DataLoader(train_data, batch_size=batch_size, random_noise=random_noise)
        test_ds = DataLoader(test_data, batch_size=batch_size, random_noise=-1)
    else:
        train_ds = DataLoader(merge_data, batch_size=batch_size, random_noise=random_noise)
    
    
    learning_rate = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=num_epochs*len(train_ds), alpha=0.0)
    model = MLP(ACT_DIM)
    print(model.tabulate(init_rng, jnp.ones([1, OBS_DIM+int(ADD_TASK_BIT)]),compute_flops=True))
    state = create_train_state(model, init_rng, learning_rate, momentum,ADD_TASK_BIT=ADD_TASK_BIT)
    state = jax.device_put(state, device)
    del init_rng  # Must not be used anymore.
    

    metrics_history = {'train_loss': []}

    for epoch in range(num_epochs):
        for step,batch in enumerate(train_ds):
            # Run optimization steps over training batches and compute batch metrics
            state = train_step(state, batch) # get updated train state (which contains the updated parameters)
            state = compute_metrics(state=state, batch=batch) # aggregate batch metrics

        for metric,value in state.metrics.compute().items(): # compute metrics
            metrics_history[f'train_{metric}'].append(value) # record metrics
        # state = state.replace(metrics=Metrics.empty())
        state = state.replace(metrics=state.metrics.empty()) 

        print(f"train epoch\t: {epoch}, "
            f"\tloss: {metrics_history['train_loss'][-1]}, ")
        
        if TRAIN_TEST_SPLIT:
            for step,batch in enumerate(test_ds):
                state = compute_metrics(state=state, batch=batch)
            for metric,value in state.metrics.compute().items():
                metrics_history[f'test_{metric}'] = value
            state = state.replace(metrics=state.metrics.empty()) 
            print(f"test epoch\t: {epoch}, "
                f"\tloss: {metrics_history['test_loss']}, ")
        
    # Save the model
    checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR,
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
        print("DEBUG: loss", loss)
        
    if TEST_AFTER_TRAINING:
        agent = MLPAgent(state, OBS_DIM, ACT_DIM, ADD_TASK_BIT)
        res = eval_agent_fast(agent, eval_episodes=100,seed=SEED, method='mp')
        params = ("Params: ", {
            'random_noise': random_noise,
            'ADD_TASK_BIT': ADD_TASK_BIT,
            'USE_DATASET_STR': USE_DATASET_STR,
            'SAMPLE_EXAMPLE': SAMPLE_EXAMPLE,
            'TRAIN_TEST_SPLIT': TRAIN_TEST_SPLIT,
            'TEST_AFTER_TRAINING': TEST_AFTER_TRAINING,
            'epoch': num_epochs,
            'batch_size': batch_size,
        })
        return (res, params)

def test(ADD_TASK_BIT=True,
         SEED=default_seed,
         ):
    np.random.seed(SEED)
    init_rng = jax.random.key(SEED)

    learning_rate = 0.005
    momentum = 0.9
    
    model = MLP(ACT_DIM)
    state = create_train_state(model, init_rng, learning_rate, momentum)
    state = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(PWD, 'ckpt','il'), target=state)
    state = jax.device_put(state, device)
    # inference
    # act = state.apply_fn({'params': state.params}, batch['obs'])
    agent = MLPAgent(state, OBS_DIM, ACT_DIM,ADD_TASK_BIT)
    eval_agent_fast(agent, eval_episodes=100,seed=SEED, method='naive')
    
def vis():
    SEED=1
    ADD_TASK_BIT=True
    np.random.seed(SEED)
    init_rng = jax.random.key(SEED)

    learning_rate = 0.005
    momentum = 0.9
    
    model = MLP(ACT_DIM)
    state = create_train_state(model, init_rng, learning_rate, momentum,ADD_TASK_BIT=ADD_TASK_BIT)
    state = checkpoints.restore_checkpoint(ckpt_dir="/home/wjxie/wjxie/env/offline_multitask/ckpt/il/20240529-092224/checkpoint_0"
                                           , target=state)
    state = jax.device_put(state, device)
    # inference
    # act = state.apply_fn({'params': state.params}, batch['obs'])
    agent = MLPAgent(state, OBS_DIM, ACT_DIM,ADD_TASK_BIT)
    
    
    env = dmc.make('walker_walk', seed=0)
    from pathlib import Path
    from UTDS.video import VideoRecorder
    from UTDS import utils
    video_recorder = VideoRecorder((Path.cwd()))   
    def eval(global_step, agent, env, logger, num_eval_episodes, video_recorder):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(num_eval_episodes)
        while eval_until_episode(episode):
            time_step = env.reset()
            video_recorder.init(env, enabled=(episode == 0))
            while not time_step.last():
                action = agent.act(time_step.observation)
                time_step = env.step(action)
                video_recorder.record(env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            video_recorder.save(f'{global_step}.mp4')
            
        print('episode_reward', total_reward / episode)
        print('episode_length', step / episode)
        print('step', global_step)
    eval(0, agent, env, None, 10, video_recorder)
    
        
if __name__=="__main__":
    smart_run({
            'train': main,
            'test': test,
            'vis': vis,
        },log_dir=CKPT_DIR)