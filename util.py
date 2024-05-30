
import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')
    
from typing import Any, List, Union, Callable
from typing import Any, Callable, Dict, Optional, Union
import dmc
import glob
import numpy as np
import gymnasium as gym
import contextlib
import fire
import os
from copy import deepcopy
import jax
import tianshou
from jax import numpy as jnp
import time
import sys
OBS_DIM = 24
ACT_DIM = 6
RUN_BIT = 1
WALK_BIT = 0
class Timer():
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
    
class DataLoader:
    def __init__(self, data:dict, batch_size=32, random_noise=-1, device=None):
        self.data = data
        for key in self.data.keys():
            self.data[key] = jax.device_put(jnp.array(data[key]), device=device)
        self.data_len = len(data['obs'])
        self.batch_size = batch_size
        self.idx = 0
        self.random_noise = random_noise
        
        self.device = device
            # -1 (or other negative) for no noise
            # 0.01 for 1% noise
        self.__shuffle()
        
    def __shuffle(self):
        idx = np.random.permutation(self.data_len)
        for key in self.data.keys():
            self.data[key] = self.data[key][idx]
    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.data_len:
            self.idx = 0
            self.__shuffle()        
            raise StopIteration
        ret = {}
        for key in self.data.keys():
            ret[key] = self.data[key][self.idx:self.idx + self.batch_size]
            if self.random_noise > 0:
                ret[key] = ret[key] + np.random.normal(0, self.random_noise, ret[key].shape)
        self.idx += self.batch_size
        return ret

    def __len__(self):
        return self.data_len // self.batch_size

class Gym2gymnasium(gym.Env):
  def __init__(self, env):
    super().__init__()
    self.env = env
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float64)
    self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    

  def __getattr__(self, __name: str) -> Any:
    return getattr(self.env, __name)
  
  def reset(self, **kwargs) -> Any:
    return self.env.reset(**kwargs)
  def step(self, action):
    return self.env.step(action)
  def render(self):
    raise NotImplementedError
    # self.env.render()

class TSPolicyAgent:
    def __init__(self, policy: tianshou.policy.BasePolicy, 
                 state_dim, action_dim,
                 ADD_TASK_BIT=False):
        self.policy = policy
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ADD_TASK_BIT = ADD_TASK_BIT
        self.task_bit = 1
    def set_task_bit(self, task_bit):
        print("DEBUG: set_task_bit", task_bit)
        self.task_bit = task_bit
        
    def act(self, state):
        assert state.shape == (self.state_dim,)
        state = np.expand_dims(state, axis=0)
        return self.act_vec(state)[0]
    def act_vec(self, states):
        assert states.shape[1] == self.state_dim
        if self.ADD_TASK_BIT:
            states = np.concatenate([states, np.ones((states.shape[0],1))*self.task_bit], axis=1)
        actions = self.policy.map_action(
            self.policy(tianshou.data.Batch(obs=states, info={}) ).act)
        return actions.detach().cpu().numpy()
    def load(self, load_path):
        # TODO:
        raise NotImplementedError
        pass
    

class TSOfflineTrainer(tianshou.trainer.OfflineTrainer):
    def __init__(self,*args, **kwargs):
        if 'eval_fn' in kwargs:
            assert 'eval_fn' in kwargs, "eval_fn should be in kwargs"
            assert 'eval_every_epoch' in kwargs, "eval_every_epoch should be in kwargs"
            self.eval_in_train = True
            self.eval_every_epoch = kwargs.pop('eval_every_epoch')
            self.eval_fn = kwargs.pop('eval_fn')
        else:
            self.eval_in_train = False
        if 'random_noise' in kwargs:
            self.random_noise = kwargs.pop('random_noise')
            self.add_noise = (self.random_noise > 0)
            # print("Debug: add noise")
        else:
            self.add_noise = False
            
        super().__init__(*args, **kwargs)
        if self.add_noise:
            self._policy_origin_process_fn = self.policy.process_fn
            
            def _process_fn_with_noise(batch: tianshou.data.Batch, buffer: tianshou.data.ReplayBuffer, indices: np.ndarray) -> tianshou.data.Batch:
                # print("Debug: add noise")
                batch = self._policy_origin_process_fn(batch, buffer, indices)
                for key in batch.keys():
                    if key in ['obs', 'obs_next']:
                        batch[key] = batch[key] + np.random.normal(0, self.random_noise, batch[key].shape)
                return batch
            
            self.policy.process_fn = _process_fn_with_noise
            
    
    def __next__(self):
        super().__next__()
        if self.eval_in_train and \
            (self.epoch % self.eval_every_epoch == 0) :
            self.eval_fn(self)

def get_ts_eval_fn(seed=1, ADD_TASKBIT=True,logger: Optional[tianshou.utils.BaseLogger] = None):
    agent = TSPolicyAgent(None, OBS_DIM, ACT_DIM,ADD_TASK_BIT=ADD_TASKBIT)
    def eval_fn(self: TSOfflineTrainer):
        # Run policy in walk and run envs   
        # Report mean and std of rewards
        agent.policy = self.policy
        # self.policy.eval()
        res = eval_agent_fast(agent, eval_episodes=10,seed=seed)
        if logger is not None:
            logger.write( "eval",self.epoch, {
                "walker_walk_mean": res["walker_walk"][0],
                "walker_run_mean": res["walker_run"][0],
                "walker_walk_std": res["walker_walk"][1],
                "walker_run_std": res["walker_run"][1],
            })
        ...
    return eval_fn

@contextlib.contextmanager
def log_and_print(log_dir='.'):
    if log_dir is None:
        yield
        return
    
    old_print = sys.stdout.write
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_handle = open(os.path.join(log_dir,'log.txt'), 'a')
    def _log_print(*args, **kwargs):
        log_handle.write(*args, **kwargs)
        old_print(*args, **kwargs)
        # sys.stdout.flush()
        log_handle.flush()
    sys.stdout.write = _log_print
    try:
        yield
    finally:
        sys.stdout.write = old_print
        log_handle.close()

def smart_run(cmd_dict: Union[dict, Callable],**kwargs):
    # supported kwargs:
        # log_dir: str
        # fire: bool
    try:
        log_dir = kwargs.get('log_dir', None)
        use_fire = kwargs.get('fire', True)
        assert use_fire or not isinstance(cmd_dict, dict), "cmd_dict should be a callable if fire is False"
        with log_and_print(log_dir):
            args = sys.argv[1:]
            print("-----"*20)
            print("Args: ", args)
            print("Log at: ", log_dir)
            print("-----"*20)
            start_time = time.time()
            if use_fire:
                ret= fire.Fire(cmd_dict)
            else:
                ret = cmd_dict()
            end_time = time.time()
            print("-----"*20)
            print("Result: ",ret)
            print("Args: ", args)
            print("Log at: ", log_dir)
            print("Program Running time: ", end_time-start_time)
    except Exception as e:
        print(">>BUG: ",e)
        import pdb;pdb.post_mortem()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def load_data(data_path):
    """
    An example function to load the episodes in the 'data_path'.
    Return: List of episodes
        [episode1, episode2, ...] 51 epsiodes
        episode1: dict_keys(['observation', 'action', 'reward', 'discount', 'physics'])
                    [(1001, 24), (1001, 6), (1001, 1), (1001, 1), (1001, 18)]
    """
    epss = sorted(glob.glob(f'{data_path}/*.npz'))
    episodes = []
    for eps in epss:
        with open(eps, 'rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
            episodes.append(episode)
    # print(len(episodes))
    # print(episodes[0].keys())
    # print([(key, val.shape) for key,val in episodes[0].items()])
    return episodes

def make_dataset(MAKE_SARSA=False, ADD_TASKBIT=True,DEAL_LAST="repeat"):
    
    run_m = load_data("collected_data/walker_run-td3-medium/data")
    run_mr = load_data("collected_data/walker_run-td3-medium-replay/data")
    walk_m = load_data("collected_data/walker_walk-td3-medium/data")
    walk_mr = load_data("collected_data/walker_walk-td3-medium-replay/data")
    # print(list(run_m[0]['physics']))
    # print(list(run_m[0]['observation']))
    
    # assert forall dataset forall traj, discount==1
    # for dataset in [run_m, run_mr, walk_m, walk_mr]:
    #     for traj in dataset:
    #         assert np.all(traj['discount'] == 1)
    
    # Reorganize them to batched dataset
    def reorganize(dataset, bit):
        print("Mean Cumulative reward: ", np.mean([np.sum(traj['reward']) for traj in dataset]))

            
            
        obs = np.concatenate([traj['observation'] for traj in dataset], axis=0)
        act = np.concatenate([traj['action'] for traj in dataset], axis=0)
        rew = np.concatenate([traj['reward'] for traj in dataset], axis=0)
        phy = np.concatenate([traj['physics'] for traj in dataset], axis=0)
        if ADD_TASKBIT:
            obs = np.concatenate([obs, np.ones((obs.shape[0],1))*bit], axis=1)
        dones = np.zeros_like(rew,dtype=np.int64)
        dones[-1] = 1
        if not MAKE_SARSA:
            return {'obs': obs, 'act': act, 'rew':rew, 'phy': phy, 'dones':dones}
        else:
        #     # FIXME:
        #     # Here we pad the last obs and act with zeros
        #     #   which may not be the best way to handle the last obs and act
            if DEAL_LAST=="pad":
                obs_prime = np.concatenate([ 
                                            np.concatenate([traj['observation'][1:], 
                                                np.zeros_like(traj['observation'][:1])]
                                                        ,axis=0) for traj in dataset], axis=0)
                act_prime = np.concatenate([ 
                                            np.concatenate([traj['action'][1:], 
                                                np.zeros_like(traj['action'][:1])]
                                                        ,axis=0) for traj in dataset], axis=0)
            elif DEAL_LAST=="repeat":
                # repeat last
                obs_prime = np.concatenate([ 
                                            np.concatenate([traj['observation'][1:], 
                                                traj['observation'][-1:]]
                                                        ,axis=0) for traj in dataset], axis=0)
                act_prime = np.concatenate([
                                            np.concatenate([traj['action'][1:], 
                                                traj['action'][-1:]]
                                                        ,axis=0) for traj in dataset], axis=0)
            elif DEAL_LAST=="remove":
                obs_prime = np.concatenate([
                                            traj['observation'][1:] for traj in dataset],axis=0)
                act_prime = np.concatenate([traj['action'][1:] for traj in dataset],axis=0)

                obs = np.concatenate([traj['observation'][:-1] for traj in dataset], axis=0)
                act = np.concatenate([traj['action'][:-1] for traj in dataset], axis=0)

                if ADD_TASKBIT:
                    obs = np.concatenate([obs, np.ones((obs.shape[0],1))*bit], axis=1)
            else:
                raise NotImplementedError



            if ADD_TASKBIT:
                obs_prime = np.concatenate([obs_prime, np.ones((obs_prime.shape[0],1))*bit], axis=1)
            
        #     obs_prime = np.concatenate([obs[1:], np.zeros_like(obs[:1])], axis=0)
        #     act_prime = np.concatenate([act[1:], np.zeros_like(act[:1])], axis=0)
            return {'obs': obs, 'act': act, 'rew':rew, 'dones':dones,
                    'obs_prime': obs_prime, 'act_prime': act_prime}
    
    run_m_data = reorganize(run_m, RUN_BIT)
    run_mr_data = reorganize(run_mr, RUN_BIT)
    walk_m_data = reorganize(walk_m, WALK_BIT)
    walk_mr_data = reorganize(walk_mr, WALK_BIT)
    
    # print( run_m_data['obs'].shape, run_m_data['act'].shape, run_m_data['phy'].shape)
    print([value.shape for value in run_m_data.values()])
    return {
        'run_m': run_m_data,
        'run_mr': run_mr_data,
        'walk_m': walk_m_data,
        'walk_mr': walk_mr_data,
    }
    
def merge_dataset(*args):
    # args: list of datasets
    # dataset: dict_keys(['obs', 'act', 'phy'])
    ret = {}
    for key in args[0].keys():
        ret[key] = np.concatenate([dataset[key] for dataset in args], axis=0)
    return ret

def make_merge_dataset(USE_DATASET: List[str]=["__all__"], MAKE_SARSA=False, ADD_TASKBIT=True, DEAL_LAST="repeat"):
    data = make_dataset(MAKE_SARSA=MAKE_SARSA, ADD_TASKBIT=ADD_TASKBIT, DEAL_LAST=DEAL_LAST)
    if "__all__" in USE_DATASET:
        USE_DATASET = list(data.keys())
    merge_data = merge_dataset(*[data[key] for key in USE_DATASET])
    return merge_data

def load_buffer_dataset(*args, **kwargs):
    # data = make_dataset(MAKE_SARSA=True,ADD_TASKBIT=ADD_TASKBIT)
    # merge_data = merge_dataset(*data.values())
    merge_data = make_merge_dataset(*args, **kwargs)
    # merge_data = merge_dataset(data['run_mr'], data['run_m'])
    # merge_data = merge_dataset(data['walk_mr'], data['walk_m'])
    merge_data['rew'] = merge_data['rew'].flatten()
    merge_data['done'] = merge_data.pop('dones').flatten() 
    merge_data['terminated'] = merge_data['done']
    merge_data['truncated'] = np.zeros_like(merge_data['done'])
    merge_data['obs_next'] = merge_data.pop('obs_prime')
    
    merge_data.pop('act_prime')
    print(merge_data.keys())
    print([v.shape for v in merge_data.values()])
    buffer = tianshou.data.ReplayBuffer.from_data(**merge_data)
    return buffer

def reward_normalize(dataset):
    # dataset: dict_keys(['obs', 'act', 'phy', 'rew'])
    rew = dataset['rew']
    mean,std = np.mean(rew), np.std(rew)
    rew = (rew - mean) / std
    assert np.abs(np.mean(rew))<1e-6 and np.abs(np.std(rew)-1)<1e-6, (np.mean(rew), np.std(rew))
    dataset['rew'] = rew
    return dataset, mean, std



def train_test_split(data, test_size=0.2):
    n = data['obs'].shape[0]
    idx = np.random.permutation(n)
    n_test = int(n*test_size)
    n_train = n - n_test
    train_data = {}
    test_data = {}
    for key in data.keys():
        train_data[key] = data[key][idx[:n_train]]
        test_data[key] = data[key][idx[n_train:]]
    return train_data, test_data

def get_gym_env(task_name, seed=1, ADD_TASKBIT=True):
    import dmc2gym
    assert task_name in ["walk","run"], f"task_name should be 'walk' or 'run', but got {task_name}"

    env = dmc2gym.make(domain_name='walker', task_name=task_name, seed=seed,
                       environment_kwargs={
                           "ADD_TASKBIT": ADD_TASKBIT,
                       })
    return env

def get_gymnasium_env(task_name, seed=1, ADD_TASKBIT=True):
    env = get_gym_env(task_name, seed=seed, ADD_TASKBIT=ADD_TASKBIT)
    env = Gym2gymnasium(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    return env


class EnvProcess(mp.Process):
    def __init__(self, task_name, seed):
        super(EnvProcess, self).__init__()
        self.task_name = task_name
        self.seed = seed
        self.conn, self.child_conn = mp.Pipe()

    def run(self):
        # print("EnvProcess: run, seed=", self.seed)
        eval_env = dmc.make(self.task_name, seed=self.seed)
        while True:
            cmd, args = self.child_conn.recv()
            # print("EnvProcess: received", cmd, args)
            if cmd =='reset':
                time_step = eval_env.reset()
                self.child_conn.send(time_step)
            elif cmd =='step':
                action = args
                time_step = eval_env.step(action)
                self.child_conn.send(time_step)
            elif cmd =='close':
                self.child_conn.close()
                break
            else:
                raise ValueError(f"Unknown command {cmd}")
    

def eval_agent_fast(agent, eval_episodes=100,seed=1, num_processes=10, method='mp'):
    assert method in ['mp', 'naive'], f"method should be 'mp' or 'naive', but got {method}"
    print('--'*10+"Start evaluation"+'--'*10)
    num_processes = min(num_processes, eval_episodes)
    HAS_ACT_VEC = hasattr(agent, 'act_vec')
    if not HAS_ACT_VEC:
        print("Warning: agent does not have act_vec method, will use act method instead")

    def evaluate_env_parallel(task_name, agent, seed):
        assert eval_episodes % num_processes == 0, f"eval_episodes should be divisible by num_processes, but got {eval_episodes} and {num_processes}"
        # num_processes = eval_episodes
        total_reward = []
            # expected shape: (eval_episodes,)
            # consider the max process number, we'd like to divide the episodes into multiple runs
            
        processes = [EnvProcess(task_name, seed+i) for i in range(num_processes)]
        for p in processes:
            # print("EnvProcess: send start")
            p.start()
            # print("EnvProcess: end  start")
        conns = [p.conn for p in processes]
        
        for i in range(0, eval_episodes, num_processes):

            [conn.send(('reset', None)) for conn in conns]
            time_steps = [conn.recv() for conn in conns]
            cumulative_reward = [0 for _ in range(num_processes)]

            while not np.all(np.stack([time_step.last() for time_step in time_steps])):
                obs_vec = np.stack([time_step.observation for time_step in time_steps])
                if HAS_ACT_VEC:
                    act_vec = agent.act_vec(obs_vec)
                else:
                    act_vec = [agent.act(obs) for obs in obs_vec]

                [conn.send(('step', action)) for conn, action in zip(conns, act_vec)]
                time_steps = [conn.recv() for conn in conns]
                cumulative_reward = [cumulative_reward[i] + time_step.reward for i, time_step in enumerate(time_steps)]
                
            total_reward.extend(cumulative_reward)
                
        [(conn.send(('close', None)), conn.close()) for conn in conns]
        [p.join() for p in processes]
            
        assert len(total_reward) == eval_episodes
        return total_reward
    
    def evaluate_env_naive(task_name, agent, seed):
        eval_envs = [dmc.make(task_name, seed=seed+i) for i in range(eval_episodes)]
        time_steps = [eval_env.reset() for eval_env in eval_envs]
        cumulative_reward = [0 for _ in range(eval_episodes)]
        while not np.all(np.stack([time_step.last() for time_step in time_steps])):
            obs_vec = np.stack([time_step.observation for time_step in time_steps])
            # shape: (eval_episodes, obs_dim)
            # action = agent.act(time_step.observation)
            if HAS_ACT_VEC:
                act_vec = agent.act_vec(obs_vec)
            else:
                act_vec = [agent.act(obs) for obs in obs_vec]
            
            time_steps = [eval_env.step(act) for eval_env, act in zip(eval_envs, act_vec)]
            cumulative_reward = [cumulative_reward[i] + time_step.reward for i, time_step in enumerate(time_steps)]
        return cumulative_reward

    def eval_agent_parallel():
        scores = {}
        for i, task_name in enumerate(["walker_walk", "walker_run"]):
            if agent.__class__.__name__ != 'Agent':
                if "run" in task_name:
                    task_bit = RUN_BIT
                else:
                    task_bit = WALK_BIT
                agent.set_task_bit(task_bit)

            # Use Ray to execute each evaluation episode in a separate process
            if method == 'mp':
                score_res = evaluate_env_parallel(task_name,agent,seed)
            else:
                score_res = evaluate_env_naive(task_name,agent,seed)
            mean = sum(score_res) / eval_episodes
            std = np.std(score_res)
            scores[task_name] = (mean, std)
            
            print(f"Task:\t {task_name}, Total Score: {score_res}")

        for task_name in ["walker_walk", "walker_run"]:
            # print(f"Task: {task_name}, Score: {scores[task_name]}")
            print(f"Task:\t {task_name}, {scores[task_name][0]} +- {scores[task_name][1]}")
        return scores
    
    res = eval_agent_parallel()
    print('--'*10+"End evaluation"+'--'*10)
    return res

def eval_agent(agent, eval_episodes=100,seed=1):
# Agent(24, 6)
    def evaluate_env(eval_env, agent, eval_episodes):
        """
        An example function to conduct online evaluation for some agentin eval_env.
        """
        returns = []
        for episode in range(eval_episodes):
            # print(f"Episode {episode}")
            time_step = eval_env.reset()
            cumulative_reward = 0
            while not time_step.last():
                action = agent.act(time_step.observation)
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

