# import gym
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from dm_control import suite

# Load the environment
# env = suite.load(domain_name="walker", task_name="run")

# Convert the dm_control environment into a Gym environment
env = gym.make('Walker2d-v4')
# env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
# try:
#   # env = make_vec_env(env, n_envs=32)
#   env = gym.make_vec('Walker2d-v4', num_envs=32)
# except Exception as e:
#   print("Error: ", e)
#   import pdb;pdb.post_mortem()

# Initialize the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000, progress_bar=True)

# Save the agent
model.save("ppo/run")

# Test the trained agent
# obs = env.reset()
obs, info = env.reset()
sum_reward = 0
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()
    sum_reward += reward
print(sum_reward)