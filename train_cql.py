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


class CQLAgent:
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


def train():
    ...
    
def test():
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