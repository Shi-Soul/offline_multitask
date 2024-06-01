
    
import argparse
import datetime
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import pprint
import gymnasium as gym
import numpy as np
from util import *
import torch


default_seed=1
PWD = os.path.dirname(os.path.abspath(__file__))
ind = time.strftime("%Y%m%d-%H%M%S")
ADD_TASKBIT = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CKPT_NAME = os.path.join('ckpt','ts_cql_final',ind)
# CKPT_DIR = os.path.join(PWD, CKPT_NAME)

def _get_cql_policy(Model_Path="ckpt/ts_cql_final/20240530-152022/policy.pth"):

    from tianshou.data import Collector
    from tianshou.env import SubprocVectorEnv
    from tianshou.policy import CQLPolicy
    from tianshou.trainer import offline_trainer
    from tianshou.utils import TensorboardLogger, WandbLogger
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.continuous import ActorProb, Critic
    
    # ADD_TASKBIT = ADD_TASKBIT
    seed = default_seed
    hidden_sizes=[256, 256]
    actor_lr = 1e-4
    critic_lr = 3e-4
    alpha = 0.2
    auto_alpha = True
    alpha_lr = 1e-4
    cql_alpha_lr = 3e-4
    tau = 0.005
    temperature = 1.0
    cql_weight = 1.0
    with_lagrange = True
    lagrange_threshold = 10.0
    gamma = 0.99
    
    
    env = get_gym_env("walk",seed=default_seed,ADD_TASKBIT=ADD_TASKBIT)
    state_shape = env.observation_space.shape or env.observation_space.n
    # args.state_shape = (args.state_shape[0]+1,)
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]  # float
    print("device:", DEVICE)
    print("Observations shape:", state_shape)
    print("Actions shape:", action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    state_dim = state_shape[0]
    action_dim = action_shape[0]
    print("Max_action", max_action)
    # replay_buffer = load_buffer_dataset()
    
    # replay_buffer = load_buffer_dataset(USE_DATASET=args.USE_DATASET_STR.split(','),
                                        # ADD_TASKBIT=ADD_TASKBIT,
                                        # MAKE_SARSA=True)

    # test_envs = SubprocVectorEnv(
    #     [lambda: get_gym_env(args.task,ADD_TASKBIT=ADD_TASKBIT) for _ in range(args.test_num)]
    # )
    # seed
    print("Seed: ",seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # test_envs.seed(seed)

    # model
    # actor network
    net_a = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        device=DEVICE,
    )
    actor = ActorProb(
        net_a,
        action_shape=action_shape,
        max_action=max_action,
        device=DEVICE,
        unbounded=True,
        conditioned_sigma=True
    ).to(DEVICE)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

    # critic network
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=DEVICE,
    )
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=DEVICE,
    )
    critic1 = Critic(net_c1, device=DEVICE).to(DEVICE)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2 = Critic(net_c2, device=DEVICE).to(DEVICE)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    if auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

    policy = CQLPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        cql_alpha_lr=cql_alpha_lr,
        cql_weight=cql_weight,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        temperature=temperature,
        with_lagrange=with_lagrange,
        lagrange_threshold=lagrange_threshold,
        min_action=np.min(env.action_space.low),
        max_action=np.max(env.action_space.high),
        device=DEVICE,
    )
    
    policy.load_state_dict(
        torch.load(Model_Path, map_location=torch.device("cpu"))
    )
    policy.eval()
    
    # result = eval_agent_fast(TSPolicyAgent(policy,
    #                                         state_dim-ADD_TASKBIT,
    #                                         action_dim,
    #                                         ADD_TASK_BIT=ADD_TASKBIT
    #                                         ),
    #                          eval_episodes=100,seed=seed, method='mp')
    
    del env 
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
                with torch.no_grad():
                # with torch.no_grad(), utils.eval_mode(agent):
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
    eval(0, TSPolicyAgent(policy,
                                            state_dim-ADD_TASKBIT,
                                            action_dim,
                                            ADD_TASK_BIT=ADD_TASKBIT
                                            ), env, None, 10, video_recorder)
    
    # return result


if __name__ == '__main__':
    smart_run({
        'cql': _get_cql_policy,
    })
    # result = _get_cql_policy()
    # print("Std Length: ", np.std(result['lens']))
    