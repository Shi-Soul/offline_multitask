
from util import *
import dmc
import numpy as np

class Agent:
    # An example of the agent to be implemented.
    # Your agent must extend from this class (you may add any other functions if needed).
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
    def act(self, state):
        assert state.shape == (self.state_dim,)
        action = np.random.uniform(-5, 5, size=(self.action_dim))
        return action
    def act_vec(self, states):
        assert states.shape[1] == self.state_dim
        actions = np.random.uniform(-5, 5, size=(states.shape[0], self.action_dim))
        return actions
    def load(self, load_path):
        pass


# make_dataset()

eval_agent_fast(Agent(24, 6), eval_episodes=100,seed=1)
# eval_agent(Agent(24, 6), eval_episodes=100,seed=1)