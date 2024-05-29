#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
from typing import Tuple
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# from examples.offline.utils import load_buffer_d4rl, normalize_all_obs_in_replay_buffer
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv, VectorEnvNormObs, DummyVectorEnv
from tianshou.data import ReplayBuffer
from tianshou.utils import RunningMeanStd
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3BCPolicy
from tianshou.trainer import offline_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from util import *

default_seed=1
PWD = os.path.dirname(os.path.abspath(__file__))
ind = time.strftime("%Y%m%d-%H%M%S")
CKPT_NAME = os.path.join('ckpt','ts_td3+bc',ind)
CKPT_DIR = os.path.join(PWD, CKPT_NAME)

def normalize_all_obs_in_replay_buffer(
    replay_buffer: ReplayBuffer
) -> Tuple[ReplayBuffer, RunningMeanStd]:
    # compute obs mean and var
    obs_rms = RunningMeanStd()
    obs_rms.update(replay_buffer.obs)
    _eps = np.finfo(np.float32).eps.item()
    # normalize obs
    replay_buffer._meta["obs"] = (replay_buffer.obs -
                                  obs_rms.mean) / np.sqrt(obs_rms.var + _eps)
    replay_buffer._meta["obs_next"] = (replay_buffer.obs_next -
                                       obs_rms.mean) / np.sqrt(obs_rms.var + _eps)
    print("DEBUG: ",obs_rms.mean, obs_rms.var, obs_rms.count)
    return replay_buffer, obs_rms

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument(
    #     "--expert-data-task", type=str, default="halfcheetah-expert-v2"
    # )
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--step-per-epoch", type=int, default=2000)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--exploration-noise", type=float, default=0.03)
    parser.add_argument("--policy-noise", type=float, default=0.1)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--norm-obs", type=int, default=0)

    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=5)
    parser.add_argument("--logdir", type=str, default="tslog")
    parser.add_argument("--render", type=float, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb","none"],
    )
    parser.add_argument("--wandb-project", type=str, default="rlp_td3bc")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    
    parser.add_argument(
        "--ADD_TASKBIT",
        type = str2bool,
        default=True,
        help="add taskbit to observation",
    )
    parser.add_argument(
        "--random_noise",
        type=float,
        default=-1,
        help="add random noise to dataset",
    )
    parser.add_argument(
        "--USE_DATASET_STR",
        type=str,
        default="__all__",
        help="use dataset string",
    )
    return parser.parse_args()


def test_td3_bc():
    args = get_args()
    ADD_TASKBIT = args.ADD_TASKBIT
    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "td3_bc"
    # log_name = os.path.join(args.algo_name, args.task, str(args.seed)+"_"+ now)
    # log_path = os.path.join(args.logdir, log_name)
    # log_name = os.path.join(args.algo_name, args.task, str(args.seed))
    # log_path = os.path.join(args.logdir, log_name)
    log_name = CKPT_NAME
    log_path = CKPT_DIR
    
    # env = gym.make(args.task)
    env = get_gym_env(args.task,seed=default_seed,ADD_TASKBIT=ADD_TASKBIT)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]  # float
    print("device:", args.device)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    args.state_dim = args.state_shape[0]
    args.action_dim = args.action_shape[0]
    print("Max_action", args.max_action)

    test_envs = SubprocVectorEnv(
        [lambda: get_gym_env(args.task,ADD_TASKBIT=ADD_TASKBIT) for _ in range(args.test_num)]
    )
    if args.norm_obs:
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        
    replay_buffer = load_buffer_dataset(USE_DATASET=args.USE_DATASET_STR.split(','),
                                        ADD_TASKBIT=ADD_TASKBIT,
                                        MAKE_SARSA=True)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)

    # model
    # actor network
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    )
    actor = Actor(
        net_a,
        action_shape=args.action_shape,
        max_action=args.max_action,
        device=args.device,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # critic network
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = TD3BCPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)


    # collector
    test_collector = Collector(policy, test_envs)


    # logger
    if not args.logger == "none":
        if args.logger == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=args.resume_id,
                config=args,
                project=args.wandb_project,
            )
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        if args.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        else:  # wandb
            logger.load(writer)
    else:
        logger = None

        

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def watch():
        if args.resume_path is None:
            args.resume_path = os.path.join(log_path, "policy.pth")

        policy.load_state_dict(
            torch.load(args.resume_path, map_location=torch.device("cpu"))
        )
        policy.eval()
        collector = Collector(policy, env)
        collector.collect(n_episode=1, render=args.render)

    if not args.watch:
        # replay_buffer = load_buffer_d4rl(args.expert_data_task)
        if args.norm_obs:
            replay_buffer, obs_rms = normalize_all_obs_in_replay_buffer(replay_buffer)
            test_envs.set_obs_rms(obs_rms)
        # trainer
        result = TSOfflineTrainer(
            policy,
            replay_buffer,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.test_num,
            args.batch_size,
            save_best_fn=save_best_fn,
            eval_fn=get_ts_eval_fn(seed=args.seed, ADD_TASKBIT=ADD_TASKBIT,logger=logger),
            eval_every_epoch=args.eval_freq,
            logger=logger or tianshou.utils.LazyLogger(),
        ).run()
        pprint.pprint(result)
    else:
        watch()

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f"Final reward: {result['rews'].mean()}, length: {result['lens'].mean()}")
    
    # return (eval_agent_fast(),args)
    result = eval_agent_fast(TSPolicyAgent(policy,
                                            args.state_dim-ADD_TASKBIT,
                                            args.action_dim,
                                            ADD_TASK_BIT=ADD_TASKBIT
                                            ),
                             eval_episodes=100,seed=args.seed, method='mp')
    return (result,args)

if __name__ == "__main__":
    
        # test_td3_bc()
    smart_run(test_td3_bc,log_dir=CKPT_DIR,fire=False)
