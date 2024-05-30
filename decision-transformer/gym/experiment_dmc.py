import gym
import numpy as np
import torch
import wandb
from tqdm import tqdm, trange

import argparse
import pickle
import random
import sys
import os
PWD = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')

import dmc
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
import dmc2gym
from util import *


default_seed=1
PWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ind = time.strftime("%Y%m%d-%H%M%S")
CKPT_NAME = os.path.join('ckpt','dt',ind)
CKPT_DIR = os.path.join(PWD, CKPT_NAME)

def make_trajs(dataset_file_paths):
    trajs = []
    for dataset_file_path in dataset_file_paths:
        for root, dirs, files in os.walk(dataset_file_path):
            for file in files:
                full_path = os.path.join(root, file)
                trajs.append(full_path)
    return trajs

def read_data(trajectories,mode, taskbit):
    states, traj_lens, returns = [], [], []
    for path_path in tqdm(trajectories):
        data = np.load(path_path)
        path = {name: data[name] for name in data.files}
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['reward'][-1] = path['reward'].sum()
            path['reward'][:-1] = 0.
        # print(path['observation'].shape)
        if taskbit:
            bit = ('run' in path_path) * 1
            states.append(np.concatenate([path['observation'], np.ones((path['observation'].shape[0], 1)) * bit], axis=1))
        else:
            states.append(path['observations'])
        # print(states[-1].shape)
        traj_lens.append(len(path['observation']))
        returns.append(path['reward'].sum())
    return states, traj_lens, returns

def from_datasetstr_to_datasetfilepath(USE_DATASET_STR):
    dataset_trans= {
        'walk_m': 'walker_walk-td3-medium',
        'walk_mr': 'walker_walk-td3-medium-replay',
        'run_m': 'walker_run-td3-medium',
        'run_mr': 'walker_run-td3-medium-replay',
    }
    if USE_DATASET_STR == '__all__':
        sublist = ['walk_m', 'walk_mr', 'run_m', 'run_mr']
    else:
        sublist = USE_DATASET_STR.split(',')
    dataset_file_paths = []
    for dataset_str in sublist:
        dataset_file_paths.append(f'../../collected_data/{dataset_trans[dataset_str]}/data')
    return dataset_file_paths
    

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    model_type = variant['model_type']
    assert model_type =='dt', 'only support dt now, bc is deprecated'
    task_bit = variant['task_bit']
    random_noise = variant['noise']
    mode = variant.get('mode', 'normal')
    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)
    
    
    group_name = f'{exp_prefix}-{variant["USE_DATASET_STR"]}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    env_walk = get_gym_env('walk',seed=1, ADD_TASKBIT=task_bit)
    env_run = get_gym_env('run',seed=1, ADD_TASKBIT=task_bit)
    max_ep_len = 1000
    env_targets = [1000, 500, 300, 200]  # evaluation conditioning targets
    scale = 1000.  # normalization for rewards/returns
    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env_walk.observation_space.shape[0]
    act_dim = env_walk.action_space.shape[0]

    # load dataset
    print('########### Loading Data ...... ###########')
    trajectories = []
    dataset_file_paths = from_datasetstr_to_datasetfilepath(variant['USE_DATASET_STR'])
    trajectories = make_trajs(dataset_file_paths)
    states, traj_lens, returns = read_data(trajectories,mode,variant['task_bit'])
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    print('########### Data Loaded! ###########')

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {variant["USE_DATASET_STR"]}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)


    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj_path = trajectories[int(sorted_inds[batch_inds[i]])]
            data = np.load(traj_path)
            traj = {name: data[name] for name in data.files}
            si = random.randint(0, traj['reward'].shape[0] - 1)

            # get sequences from dataset
            traj['dones'] = np.zeros(len(traj['observation']))
            traj['dones'][-1] = 1
            if random_noise > 0:
                traj['observation'] += np.random.normal(0, random_noise, traj['observation'].shape)
            if variant['task_bit']:
                bit = ('run' in traj_path) * 1
                traj['observation'] = np.concatenate([traj['observation'], np.ones((traj['observation'].shape[0], 1)) * bit], axis=1)
            s.append(traj['observation'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['action'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['reward'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['reward'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns_walk = []
            returns_run = []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret_walk, __ = evaluate_episode_rtg(
                        env_walk,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                    ret_run, __ = evaluate_episode_rtg(
                        env_run,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                    
                returns_walk.append(ret_walk)
                returns_run.append(ret_run)
            return {
                # f'target_{target_rew}_return_mean': np.mean(returns),
                # f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_return_walk_mean': np.mean(returns_walk),
                f'target_{target_rew}_return_walk_std': np.std(returns_walk),
                f'target_{target_rew}_return_run_mean': np.mean(returns_run),
                f'target_{target_rew}_return_run_std': np.std(returns_run),
            }
        return fn



    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    print(model)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    print('########### Begin Training ...... ###########')
    for it in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=it+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)
        if variant['save_model']:
            trainer.model.save(os.path.join(CKPT_DIR, str(it)+".pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='dmc_run')
    # parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, all
    parser.add_argument('--USE_DATASET_STR', type=str, default='__all__') 
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=2)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--task_bit', type=bool, default=True)
    parser.add_argument('--noise', type=float, default=-1)
    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
        
        
if __name__ == '__main__':
    
    smart_run(main,fire=False,log_dir=CKPT_DIR)