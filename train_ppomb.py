from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env
from dm_control import suite
from functools import partial
import fire
import time
from util import *

SEED=1
PWD = os.path.dirname(os.path.abspath(__file__))
device = jax.devices("gpu")[0]
assert device.platform=="gpu"


# Define type aliases here to avoid circular import
# Used when we want to access one or more VecEnv
VecEnvIndices = Union[None, int, Iterable[int]]
# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, ...]]
# VecEnvStepReturn is what is returned by the step() method
# it contains the observation, reward, done, info for each env
VecEnvStepReturn = Tuple[VecEnvObs, np.ndarray, np.ndarray, List[Dict]]

class MLPModelGymEnv(VecEnv):
# class MLPModelGymEnv(gym.Env):
    """ A Model Based Env
        In each episodes, 
            1. sample some random init states from dataset as the first state
            2. predict the next state and reward using the model for next N steps
            3. done at N-th step
    """
    def __init__(self, model, dataset, num_steps=100, num_envs=16):
        self.model = model
        self.dataset = dataset["obs"]
        self.N = num_envs
        self.n_pred_step = num_steps
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float64)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.render_mode = None
        super().__init__(num_envs, self.observation_space, self.action_space)
        
        self._state = None
        self._steps = 0
            # shape: (N, O)
        
    def reset(self, **kwargs) -> Any:
        # (N, O)
        # return self.env.reset(**kwargs)
        # obs, info
        self._state = self.dataset[np.random.choice(len(self.dataset),self.N)]
        self._steps = 0
        # self._state = jnp.array(self.dataset[np.random.choice(len(self.dataset), self.N)])
        return self._state
        raise NotImplementedError
    
    def step(self, action):
        # (N, A) => (N, O), (N, ), (N, ), (N, ), [{},{},...]
        # obs, reward, done, truncated, info
        self._steps +=1
        info = [{} for i in range(self.N)]
            
        model_output = self.model.apply_fn({'params': self.model.params},self._state, jnp.array(action))
        # O + 1(rewards)
        obs = model_output[:, :-1]
        self._state = obs
        reward = np.array(model_output[:, -1])
        if self._steps >= self.n_pred_step:
            done=True
        else:
            done=False
            for i in range(self.N):
                info[i]["terminal_observation"]=np.array(obs[i])
        return np.array(obs), reward, np.ones_like(reward,dtype=np.bool_)*done, info
        # return obs, reward, np.zeros_like(reward,dtype=np.bool_)*done, np.zeros_like(reward,dtype=np.bool_), {}
        # return obs.squeeze(), reward[0], done, False, {}
    
    def step_async(self, actions: np.ndarray) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        raise NotImplementedError()

    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        raise NotImplementedError()

    def close(self) -> None:
        """
        Clean up the environment's resources.
        """
        # raise NotImplementedError()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """
        Return attribute from vectorized environment.

        :param attr_name: The name of the attribute whose value to return
        :param indices: Indices of envs to get attribute from
        :return: List of values of 'attr_name' in all environments
        """
        if attr_name == "render_mode":
            return [self.render_mode]*(len(indices) if indices is not None else self.num_envs)
        raise NotImplementedError()

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """
        Set attribute inside vectorized environments.

        :param attr_name: The name of attribute to assign new value
        :param value: Value to assign to `attr_name`
        :param indices: Indices of envs to assign value
        :return:
        """
        raise NotImplementedError()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """
        Call instance methods of vectorized environments.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: List of items returned by the environment's method call
        """
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """
        Check if environments are wrapped with a given wrapper.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: True if the env is wrapped, False otherwise, for each env queried.
        """
        raise NotImplementedError()

def run_test(env,model):
    obs = env.reset()
    sum_reward = 0
    for i in range(1000):
        # print(obs,obs.shape)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # obs, reward, done, truncated, info = env.step(action)
        if done.all():
            # obs = env.reset()
            obs= env.reset()
        sum_reward += reward
    return sum_reward
        
def train():
    np.random.seed(SEED)
    init_rng = jax.random.PRNGKey(SEED)

    data = make_dataset(True,True,"remove")
    merge_data = merge_dataset(data['walk_mr'],data['walk_m'])
    from train_mbmlp import MLP, create_train_state, train_step, compute_metrics
    from flax.training import checkpoints
    envmodel = MLP(OBS_DIM+2)
    state = create_train_state(envmodel, init_rng, 2e-3, 0.99)
    state = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(PWD, 'ckpt','mb'), target=state)
    state = jax.device_put(state, device)
    
    env = MLPModelGymEnv(state, merge_data, num_steps=10, num_envs=16)
    
    realenv = make_vec_env(env_id = lambda:get_gymnasium_env("walk"), n_envs=16)
    
    # Initialize the agent
    model = PPO("MlpPolicy", env, verbose=1, n_steps=256)
    print("Before Training: The Model thinks how good the policy is ", run_test(env,model))
    print("Before Training: The Real Env thinks how good the policy is ", run_test(realenv,model))
    

    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save("ppomb/base")

    print("After  Training: The Model thinks how good the policy is ", run_test(env,model))
    print("After  Training: The Real Env thinks how good the policy is ", run_test(realenv,model))
    
    
    
        
if __name__=="__main__":
    try:
        start_time = time.time()
        fire.Fire({
            'train': train,
            # 'test': test
        })
        end_time = time.time()
        print("Program Running time: ", end_time-start_time)
    except Exception as e:
        print(">>BUG: ",e)
        import pdb;pdb.post_mortem()
