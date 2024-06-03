import numpy as np
import glob

# For each episode, we should load it, find the key named 'action' and 'reward'
# remove the first element of them, and pad 0 at the end of this episode
# Then, we should write it back to the file

def process_data(in_data_path,out_data_path):
    epss = sorted(glob.glob(f'{in_data_path}/*.npz'))
    for eps in epss:
        with open(eps, 'rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
            # print(episode.keys())
            # print([(key, val.shape) for key,val in episode.items()])
            action = episode['action']
            reward = episode['reward']
            episode['action'] = np.concatenate([action[1:],np.zeros((1, action.shape[1]))], axis=0)
            episode['reward'] = np.concatenate([reward[1:],np.zeros((1, reward.shape[1]))], axis=0)
            # print([(key, val.shape) for key,val in episode.items()])
            np.savez(f'{out_data_path}/{eps.split("/")[-1]}', **episode)
            # print(f'{out_data_path}/{eps.split("/")[-1]}')



process_data("collected_data_old/walker_run-td3-medium/data","collected_data/walker_run-td3-medium/data")
process_data("collected_data_old/walker_walk-td3-medium/data","collected_data/walker_walk-td3-medium/data")
process_data("collected_data_old/walker_run-td3-medium-replay/data","collected_data/walker_run-td3-medium-replay/data")
process_data("collected_data_old/walker_walk-td3-medium-replay/data","collected_data/walker_walk-td3-medium-replay/data")