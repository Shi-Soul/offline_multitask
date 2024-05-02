import dmc
import glob
import numpy as np

OBS_DIM = 24
ACT_DIM = 6
RUN_BIT = 1
WALK_BIT = 0


class DataLoader:
    def __init__(self, data:dict, batch_size=32, random_noise=-1):
        self.data = data
        self.data_len = len(data['obs'])
        self.batch_size = batch_size
        self.idx = 0
        self.random_noise = -1
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

def make_dataset():
    
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
        obs = np.concatenate([traj['observation'] for traj in dataset], axis=0)
        act = np.concatenate([traj['action'] for traj in dataset], axis=0)
        phy = np.concatenate([traj['physics'] for traj in dataset], axis=0)
        obs = np.concatenate([obs, np.ones((obs.shape[0],1))*bit], axis=1)
        return {'obs': obs, 'act': act, 'phy': phy}
    
    run_m_data = reorganize(run_m, RUN_BIT)
    run_mr_data = reorganize(run_mr, RUN_BIT)
    walk_m_data = reorganize(walk_m, WALK_BIT)
    walk_mr_data = reorganize(walk_mr, WALK_BIT)
    
    print( run_m_data['obs'].shape, run_m_data['act'].shape, run_m_data['phy'].shape)
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
        if agent.__class__.__name__ == 'MLPAgent':
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

