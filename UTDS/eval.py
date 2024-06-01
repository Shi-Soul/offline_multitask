import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import time 
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
import random
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import json
from pathlib import Path
# import hydra
import numpy as np
import torch
from dm_env import specs
import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder
import wandb
import yaml
from box import Box
torch.backends.cudnn.benchmark = True

import torch
import numpy as np
from UTDS.agent.cql_cds import CQLCDSAgent
from util import eval_agent_fast, eval_agent

class Eval_UTDS_Agent:
    def __init__(self, original_agent):
        self._agent = original_agent
        
    def act(self, state):
        return self._agent.act(state.reshape(1,-1))[0]
    
    def act_vec(self, states):
        # assert states.shape[1] == self.state_dim
        # actions = np.random.uniform(-5, 5, size=(states.shape[0], self.action_dim))
        with torch.no_grad():
            actions = self._agent.act(states,step=0, eval_mode=True)
        return actions#.detach().cpu().numpy()
    def set_task_bit(self,taskbit):
        ...

if __name__ == "__main__":
    # agent = UTDS_Agent(24, 6)
    with open('config_cds.yaml', 'r') as file:
        config = yaml.safe_load(file)
    cfg = Box(config)

    env = dmc.make(cfg.task, seed=cfg.seed)
    
    agent = CQLCDSAgent(obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape, num_expl_steps=0,
        name=cfg.name, actor_lr=float(cfg.actor_lr), critic_lr=float(cfg.critic_lr),
        critic_target_tau=cfg.critic_target_tau, n_samples = cfg.n_samples,
        use_critic_lagrange = cfg.use_critic_lagrange, alpha = cfg.alpha,
        target_cql_penalty = cfg.target_cql_penalty,
        use_tb = cfg.use_tb,
        hidden_dim = cfg.hidden_dim,
        nstep = cfg.nstep,
        batch_size = cfg.batch_size,
        has_next_action = cfg.has_next_action,
        device=cfg.device)
    
    agent.actor= torch.load('actor_walker_walk_1717162661.098715_1024_15000.pth', 
                            map_location=torch.device('cuda'))            
    
    # agent = Eval_UTDS_Agent(agent)
    
    # eval_agent_fast(agent, eval_episodes=100,seed=1)
    
    video_recorder = VideoRecorder((Path.cwd()))   
    from UTDS.train_offline_cds import eval
    eval(0, agent, env, None, 10, video_recorder)
    

