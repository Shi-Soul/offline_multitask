import numpy as np
import torch

class DTAgent: #TODO: Implement the agent here
    def __init__(self, model,
                    state_dim, action_dim,
                    state_mean=0., state_std=1.,
                    scale=1000.,
                    rtg=None,
                    device='cuda',
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = model
        self.state_mean = torch.from_numpy(state_mean).to(device=device)
        self.state_std = torch.from_numpy(state_std).to(device=device)
        self.scale = scale
        self.device = device
        
        self.target_return = torch.tensor(rtg, device=device, dtype=torch.float32).reshape(1, 1)
        self.timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        self.t = 0
        
        self.model.eval()
        
    def act(self, state):
        assert state.shape == (self.state_dim,)
        action = np.random.uniform(-5, 5, size=(self.action_dim))
        return action
    
    def reset_rtg(self,rtg):
        self.target_return = torch.tensor(rtg, device=self.device, dtype=torch.float32).reshape(1, 1)
        self.timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
        ...
    
    

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