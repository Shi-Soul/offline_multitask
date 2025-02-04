
#general

import argparse
import json
import subprocess
import numpy as np
from tqdm import tqdm
import os
import glob
import tarfile
from comet_ml import Experiment

from morel.morel import Morel
import torch
# import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import os
from functools import partial
import fire
import time
from util import get_gymnasium_env,make_dataset,merge_dataset, OBS_DIM, ACT_DIM

SEED=1
PWD = os.path.dirname(os.path.abspath(__file__))
# device = jax.devices("gpu")[0]
# assert device.platform=="gpu"
EPS=1e-9

class Maze2DDataset(Dataset):

    def __init__(self):
        # self.env = gym.make('maze2d-umaze-v1')
        # dataset = self.env.get_dataset()
        self.env = get_gymnasium_env("walk")
        
        data = make_dataset(False,True)
        dataset = merge_dataset(data['walk_mr'],data['walk_m'])
        # dataset = merge_dataset(data['walk_mr'],data['walk_m'])

        # Input data
        self.source_observation = dataset["obs"][:-1]
        self.source_action = dataset["act"][:-1]


        # Output data
        self.target_delta = dataset["obs"][1:] - self.source_observation
        self.target_reward = dataset["rew"][:-1]

        # Normalize data
        self.delta_mean = self.target_delta.mean(axis=0)
        self.delta_std = self.target_delta.std(axis=0)+EPS

        self.reward_mean = self.target_reward.mean(axis=0)
        self.reward_std = self.target_reward.std(axis=0)+EPS

        self.observation_mean = self.source_observation.mean(axis=0)
        self.observation_std = self.source_observation.std(axis=0)+EPS

        self.action_mean = self.source_action.mean(axis=0)
        self.action_std = self.source_action.std(axis=0)+EPS

        self.source_action = (self.source_action - self.action_mean)/self.action_std
        self.source_observation = (self.source_observation - self.observation_mean)/self.observation_std
        self.target_delta = (self.target_delta - self.delta_mean)/self.delta_std
        self.target_reward = (self.target_reward - self.reward_mean)/self.reward_std

        # Get indices of initial states
        self.done_indices = dataset["dones"][:-1]
        self.initial_indices = np.roll(self.done_indices, 1)
        self.initial_indices[0] = True

        # Calculate distribution parameters for initial states
        self.initial_obs = self.source_observation[self.initial_indices]
        self.initial_obs_mean = self.initial_obs.mean(axis = 0)
        self.initial_obs_std = self.initial_obs.std(axis = 0)

        # Remove transitions from terminal to initial states
        self.source_action = np.delete(self.source_action, self.done_indices, axis = 0)
        self.source_observation = np.delete(self.source_observation, self.done_indices, axis = 0)
        self.target_delta = np.delete(self.target_delta, self.done_indices, axis = 0)
        self.target_reward = np.delete(self.target_reward, self.done_indices, axis = 0)



    def __getitem__(self, idx):
        feed = torch.FloatTensor(np.concatenate([self.source_observation[idx], self.source_action[idx]])).to("cuda:0")
        target = torch.FloatTensor(np.concatenate([self.target_delta[idx], self.target_reward[idx]])).to("cuda:0")

        return feed, target

    def __len__(self):
        return len(self.source_observation)

def upload_assets(comet_experiment, log_dir):
    tar_path = log_dir + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(log_dir, arcname=os.path.basename(log_dir))

    comet_experiment.log_asset(tar_path)
    os.remove(tar_path)

def main(args):
    tensorboard_writer = None
    comet_experiment = None

    if(not args.no_log):
        # Create necessary directories
        if(not os.path.isdir(args.log_dir)):
            os.mkdir(args.log_dir)

        # Create log_dir for run
        run_log_dir = os.path.join(args.log_dir,args.exp_name)
        if(os.path.isdir(run_log_dir)):
            cur_count = len(glob.glob(run_log_dir + "_*"))
            run_log_dir = run_log_dir + "_" + str(cur_count)
        os.mkdir(run_log_dir)
        print("Logging to: ", run_log_dir)

        # Create tensorboard writer if requested

        if(args.tensorboard):
            tensorboard_dir = os.path.join(run_log_dir, "tensorboard")
            writer = SummaryWriter(log_dir = tensorboard_dir)


    # Create comet experiment if requested
    if(args.comet_api is not None):
        comet_experiment = Experiment(
            api_key = args.comet_api,
            project_name = "RLP_MOReL",
            workspace = "shi-soul",
        )
        comet_experiment.set_name(args.exp_name)

        # Get hash for latest git commit for logging
        last_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").rstrip()
        comet_experiment.log_parameter("git_commit_id", last_commit_hash)

    # Instantiate dataset
    dynamics_data = Maze2DDataset()

    dataloader = DataLoader(dynamics_data, batch_size=256, shuffle = True)

    agent = Morel(OBS_DIM+1, ACT_DIM, tensorboard_writer = tensorboard_writer, comet_experiment = comet_experiment)

    agent.train(dataloader, dynamics_data, load_dynamics=None)
    # agent.train(dataloader, dynamics_data, load_dynamics="/home/wjxie/wjxie/env/offline_multitask/morel/results/exp_test_24/models")

    if(not args.no_log):
        print("Save at: ", run_log_dir)
        agent.save(os.path.join(run_log_dir, "models"))
        if comet_experiment is not None:
            upload_assets(comet_experiment, run_log_dir)

    agent.eval(dynamics_data.env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')

    parser.add_argument('--log_dir', type=str, default='./morel/results/')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--comet_api', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='exp_test')
    parser.add_argument('--no_log', action='store_true')


    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print("BUG>>>>> ",e)
        import pdb;pdb.post_mortem()






