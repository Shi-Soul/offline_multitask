import numpy as np
import torch
import os 
import sys

PWD = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')
from util import *

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from experiment_dmc import from_datasetstr_to_datasetfilepath, make_trajs, read_data

class DTAgent: 
    def __init__(self, model,
                    state_dim, action_dim,
                    state_mean=0., state_std=1.,
                    scale=1000.,
                    rtg=1000,
                    device='cuda',
                    ADD_TASK_BIT=True,
                 ):
        self.model = model
        self.model.eval()
        self.mode = 'normal'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_mean = torch.from_numpy(state_mean).to(device=device)
        self.state_std = torch.from_numpy(state_std).to(device=device)
        self.scale = scale
        self.device = device
        self.ADD_TASK_BIT = ADD_TASK_BIT
        self.task_bit = RUN_BIT
        
        self.reset_rtg(rtg)
        
        
    def act(self, state, prev_rew=0 ):
        assert state.shape == (self.state_dim,)
        
        if self.ADD_TASK_BIT:
            state = np.concatenate([state, np.ones((1))*self.task_bit], axis=0)
        # action = np.random.uniform(-5, 5, size=(self.action_dim))
        if self.t==0:
            self.states = torch.from_numpy(state).reshape(1, self.state_dim+self.ADD_TASK_BIT).to(device=self.device, dtype=torch.float32)
            self.actions = torch.zeros((0, self.action_dim), device=self.device, dtype=torch.float32)
            self.rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        else:
            cur_state = torch.from_numpy(state).to(device=self.device).reshape(1, self.state_dim+self.ADD_TASK_BIT)
            self.states = torch.cat([self.states, cur_state], dim=0)
            self.rewards[-1] = prev_rew

            if self.mode != 'delayed':
                pred_return = self.target_return[0,-1] - (prev_rew/self.scale)
            else:
                pred_return = self.target_return[0,-1]
            self.target_return = torch.cat(
                [self.target_return, pred_return.reshape(1, 1)], dim=1)
            self.timesteps = torch.cat(
                [self.timesteps,
                torch.ones((1, 1), device=self.device, dtype=torch.long) * (self.t)], dim=1)
            
            
        self.actions = torch.cat([self.actions, torch.zeros((1, self.action_dim), device=self.device)], dim=0)
        self.rewards = torch.cat([self.rewards, torch.zeros(1, device=self.device)])
        
        action = self.model.get_action(
            (self.states.to(dtype=torch.float32) - self.state_mean) / self.state_std,
            self.actions.to(dtype=torch.float32),
            self.rewards.to(dtype=torch.float32),
            self.target_return.to(dtype=torch.float32),
            self.timesteps.to(dtype=torch.long),
        )
        self.actions[-1] = action
        action = action.detach().cpu().numpy()
        
        # state, reward, done, _, _ = env.step(action)
        

        self.t+=1
        
        return action
    
    def reset_rtg(self,rtg):
        self.target_return = torch.tensor(rtg, device=self.device, dtype=torch.float32).reshape(1, 1)
        self.timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
        self.t = 0
        
    def set_task_bit(self, task_bit):
        self.task_bit = task_bit
        ...

def eval_episodes(target_rew, num_eval_episodes, env_walk, env_run, state_dim, act_dim, max_ep_len, scale, state_mean, state_std, device, mode='normal'):
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


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state, _ = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _, _ = env.step(action)
        # print('step done!!!')

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

def eval_agent(agent, eval_episodes=100,seed=1, rtg=0.8):
    # import pdb;pdb.set_trace()
# Agent(24, 6)
    def evaluate_env(eval_env, agent, eval_episodes):
        """
        An example function to conduct online evaluation for some agentin eval_env.
        """
        returns = []
        for episode in range(eval_episodes):
            # print(f"Episode {episode}")
            time_step = eval_env.reset()
            agent.reset_rtg(rtg)
            cumulative_reward = 0
            while not time_step.last():
                action = agent.act(time_step.observation, time_step.reward)
                # action = agent.act(time_step.observation)
                time_step = eval_env.step(action)
                cumulative_reward += time_step.reward
            returns.append(cumulative_reward)
            print(f"Episode {episode} reward: {cumulative_reward}")
        return sum(returns) / eval_episodes

    scores = {}
    for task_name in ["walker_walk", "walker_run"]:
        # seed = 42
        # if "run" in task_name:
        if agent.__class__.__name__ != 'Agent':
            if "run" in task_name:
                task_bit = RUN_BIT
            else:
                task_bit = WALK_BIT
            agent.set_task_bit(task_bit)
            
        eval_env = dmc.make(task_name, seed=seed)
        score_walk = (evaluate_env(eval_env=eval_env, agent=agent, eval_episodes=eval_episodes))
        scores[task_name] = score_walk


    for task_name in ["walker_walk", "walker_run"]:
        print(f"Task: {task_name}, Score: {scores[task_name]}")
    return scores


def main():
    device = 'cuda'
    task_bit = 1

    env_walk = get_gym_env('walk',seed=1, ADD_TASKBIT=task_bit)
    env_run = get_gym_env('run',seed=1, ADD_TASKBIT=task_bit)
    max_ep_len = 1000
    env_targets = [0.8,1,1000, 800, 500, 300, 200]  # evaluation conditioning targets
    scale = 1000.  # normalization for rewards/returns

    state_dim = env_walk.observation_space.shape[0]
    act_dim = env_walk.action_space.shape[0]

    mode = 'normal'
    K= 20
    variant={
        'embed_dim': 256,
        'n_layer': 3,
        'n_head': 1,
        'activation_function': 'relu',
        'dropout': 0.1,
        'USE_DATASET_STR': '__all__',
        'task_bit':task_bit,
    }

        

    dt_model = DecisionTransformer(
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
            ).to(device)
    dt_model.load("/home/wjxie/wjxie/env/offline_multitask/ckpt/_dt/9.pt")
    
    dataset_file_paths = from_datasetstr_to_datasetfilepath(variant['USE_DATASET_STR'])
    trajectories = make_trajs(dataset_file_paths)
    states, traj_lens, returns = read_data(trajectories,mode,variant['task_bit'])
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    print('########### Data Loaded! ###########')

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    # print('state_mean:', state_mean)
    # print('state_std:', state_std)
        
    dt_agent = DTAgent(dt_model, state_dim-task_bit, act_dim, state_mean, state_std, scale, 1000, device)
    # print(eval_agent(dt_agent, eval_episodes=10,seed=1,))
    for rtg in env_targets:
        print(eval_agent(dt_agent, eval_episodes=10,seed=1,rtg=rtg))
        
    # for env in [env_walk, env_run]:
    #     for target_return in env_targets:
    #         print(eval_episodes(target_return, 2, env, env, 2, 2, max_ep_len, scale, 0, 1, 'cuda', mode='normal')(dt_model))
    
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("BUG>>>>> ", e)
        import pdb;pdb.post_mortem()