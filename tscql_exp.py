#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# from examples.offline.utils import load_buffer_d4rl
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.policy import CQLPolicy
from tianshou.trainer import offline_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from util import *
default_seed=1
PWD = os.path.dirname(os.path.abspath(__file__))
# ind = time.strftime("%Y%m%d-%H%M%S")

ind = time.strftime("%Y%m%d-%H%M%S")+str(np.random.randint(1000))
CKPT_NAME = os.path.join('ckpt','ts_cql_exp',ind)
CKPT_DIR = os.path.join(PWD, CKPT_NAME)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)  #alpha for SAC entropy term
    parser.add_argument("--auto-alpha", default=True, action="store_true") # auto alpha for SAC entropy term
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=2500)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)

    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cql-weight", type=float, default=50.0)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.99)

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
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="rlp_cql")
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


def test_cql():
    args = get_args()
    ADD_TASKBIT = args.ADD_TASKBIT
    # env = get_gym_env(args.task)
    env = get_gym_env(args.task,seed=default_seed,ADD_TASKBIT=ADD_TASKBIT)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    # args.state_shape = (args.state_shape[0]+1,)
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]  # float
    print("device:", args.device)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    args.state_dim = args.state_shape[0]
    args.action_dim = args.action_shape[0]
    print("Max_action", args.max_action)
    # replay_buffer = load_buffer_dataset()
    
    replay_buffer = load_buffer_dataset(USE_DATASET=args.USE_DATASET_STR.split(','),
                                        ADD_TASKBIT=ADD_TASKBIT,
                                        MAKE_SARSA=True)

    test_envs = SubprocVectorEnv(
        [lambda: get_gym_env(args.task,ADD_TASKBIT=ADD_TASKBIT) for _ in range(args.test_num)]
    )
    # seed
    print("Seed: ",args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)

    # model
    
    
    
    # actor network
    net_a = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        norm_layer=torch.nn.LayerNorm,
        activation=torch.nn.SiLU,
    )
    actor = ActorProb(
        net_a,
        action_shape=args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # critic network
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
        norm_layer=torch.nn.LayerNorm,
        activation=torch.nn.SiLU,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
        norm_layer=torch.nn.LayerNorm,
        activation=torch.nn.SiLU,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = CQLPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        cql_alpha_lr=args.cql_alpha_lr,
        cql_weight=args.cql_weight,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        temperature=args.temperature,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        min_action=np.min(env.action_space.low),
        max_action=np.max(env.action_space.high),
        device=args.device,
        clip_grad=1000.0,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    test_collector = Collector(policy, test_envs)

    # log
    log_name = CKPT_NAME
    log_path = CKPT_DIR

    # logger
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
        # collector.collect(n_episode=1, render=1 / 35)

    if not args.watch:
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
            random_noise = args.random_noise,
            logger=logger,
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

    result = eval_agent_fast(TSPolicyAgent(policy,
                                            args.state_dim-ADD_TASKBIT,
                                            args.action_dim,
                                            ADD_TASK_BIT=ADD_TASKBIT
                                            ),
                             eval_episodes=100,seed=args.seed, method='mp')
    return (result,args)

if __name__ == "__main__":
    smart_run(test_cql,log_dir=CKPT_DIR,fire=False)
    # CUDA_VISIBLE_DEVICES=2 python tianshou_cql.py --random_noise=-1 --ADD_TASKBIT=True --USE_DATASET_STR="walk_m,walk_mr" --task walk                