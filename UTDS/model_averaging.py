import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'agent')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import yaml
from box import Box
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict

# import internal libs
from cql_cds import Actor, Critic

def test_average(model_pathA: str,
                 model_pathB: str,
                 model: nn.Module,
                 save_path: str):
    """get the model average.

    Args:
        model_paths: the paths of the models to average
        model: the model
    """
    averaged_model = OrderedDict()
    model_weights = {}
    total_num = 0
    model_paths = [model_pathA, model_pathB]
    for model_path in tqdm(model_paths):
        model_weights[total_num] = torch.load(model_path, map_location="cpu").state_dict()
        for k, v in model_weights[total_num].items():
            if k.startswith("module."):
                k = k[7:]
            if total_num == 0:
                averaged_model[k] = v
            else:
                averaged_model[k] += v
        total_num += 1
    for k, v in model_weights[0].items():
        if k.startswith("module."):
            k = k[7:]
        averaged_model[k] = averaged_model[k] / total_num
    model.load_state_dict(averaged_model)

    torch.save(model, f"{save_path}")


def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="simple verification of the average model")
    ## the basic setting of exp
    parser.add_argument("--save_path", default="_walker_average_1024.pth", type=str,
                        help='the path of saving results.')
    parser.add_argument("--actor_walk_path", default="actor_walker_walk_1717166979.3499706_1024.pth", type=str,
                        help='the path of pretrained actor of task walk.')
    parser.add_argument("--actor_run_path", default="actor_walker_run_1717162567.3957305_1024.pth", type=str,
                        help='the path of pretrained actor of task run.')
    parser.add_argument("--critic_walk_path", default="critic_walker_walk_1717166979.3765106_1024.pth", type=str,
                        help='the path of pretrained critic of task walk.')
    parser.add_argument("--critic_run_path", default="critic_walker_run_1717162567.437914_1024.pth", type=str,
                        help='the path of pretrained critic of task run.')
    parser.add_argument("--cfg_path", default="config_cds.yaml", type=str,
                        help='the model name.')
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    return args


def main():
    # get the args.
    args = add_args()

    # get the configs
    print("########getting config....")
    with open(args.cfg_path, 'r') as file:
        config = yaml.safe_load(file)
    config = Box(config)
    print("getting config done!")

    # prepare the model
    print("#########preparing model....")
    state_dim = 24
    action_dim = 6
    hidden_dim = 1024
    actor = Actor(state_dim, action_dim, hidden_dim)
    critic = Critic(state_dim, action_dim, hidden_dim)
    print("preparing model done!")

    # get the average of models
    print("#########geting the average of actors....")
    test_average(model_pathA = args.actor_walk_path,
                 model_pathB = args.actor_run_path,
                 model = actor,
                 save_path = 'actor' + args.save_path)
    
    test_average(model_pathA = args.critic_walk_path,
                 model_pathB = args.critic_run_path,
                 model = critic,
                 save_path = 'critic' + args.save_path)


if __name__ == "__main__":
    main()