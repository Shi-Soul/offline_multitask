import glob 
import numpy as np

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
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
    return episodes

def vis(env,episode,video_recorder, name = "0"):
    states = episode['physics']
    video_recorder.init(env, enabled=True)
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        video_recorder.record(env)
    video_recorder.save(f'{name}.mp4')

import dmc 
env = dmc.make('walker_walk', seed=0)
from pathlib import Path
from UTDS.video import VideoRecorder
from UTDS import utils
video_recorder = VideoRecorder((Path.cwd()))  

# data = load_data("collected_data/walker_walk-td3-medium/data")
# 
# for episode in data:
#     vis(env,episode,video_recorder)
#     break
vis(env,load_data("collected_data/walker_walk-td3-medium/data")[0],video_recorder, name = "walk_m")
vis(env,load_data("collected_data/walker_walk-td3-medium-replay/data")[1],video_recorder, name = "walk_mr")
vis(env,load_data("collected_data/walker_run-td3-medium/data")[0],video_recorder, name = "run_m")
vis(env,load_data("collected_data/walker_run-td3-medium-replay/data")[1],video_recorder, name = "run_mr")

