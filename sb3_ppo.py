# import gym
from typing import Any
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from dm_control import suite
from util import *

def main():
  # Load the environment
  # env = suite.load(domain_name="walker", task_name="run")

  # Convert the dm_control environment into a Gym environment
  # env = gym.make('Walker2d-v4')
  env = get_gymnasium_env("walk")
  # env = get_gym_env("walk")
  # env = Gym2gymnasium(env)
  # env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
  # env = make_vec_env(env_id = lambda:get_gymnasium_env("walk"), n_envs=16)
  # env = gym.make_vec('Walker2d-v4', num_envs=32)

  # Initialize the agent
  model = PPO("MlpPolicy", env, verbose=1)

  # Train the agent
  model.learn(total_timesteps=500000, progress_bar=True)

  # Save the agent
  model.save("ppo/run")

  # Test the trained agent
  # obs = env.reset()
  obs, info = env.reset()
  sum_reward = 0
  for i in range(1000):
      # print(obs,obs.shape)
      action, _states = model.predict(obs, deterministic=True)
      # obs, reward, done, info = env.step(action)
      obs, reward, done, truncated, info = env.step(action)
      if done:
        # obs = env.reset()
        obs, info = env.reset()
      sum_reward += reward
  print(sum_reward)
  
if __name__ == "__main__":
  try :
    main()
  except Exception as e:
    print(f"Error: {e}")
    import pdb;pdb.post_mortem()