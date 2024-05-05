import jax
import jax.numpy as jnp
from flax.core import FrozenDict
import flax.linen as nn
from jax import random
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax.training import checkpoints
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers
import numpy as np
from functools import partial
import fire
import time
from util import *
import os

SEED=1
PWD = os.path.dirname(os.path.abspath(__file__))
device = jax.devices("gpu")[0]
assert device.platform=="gpu"


class CQLSACAgent:
    def __init__(self, modelstate, state_dim, action_dim, rng=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.modelstate = modelstate
        self.task_bit = 1
        
        self.rng = rng
        
    def set_task_bit(self, task_bit):
        self.task_bit = task_bit
        
    def act(self, state):
        assert state.shape == (self.state_dim,)
        self.rng, rng = random.split(self.rng)
        state = np.concatenate([state, np.ones((1))*self.task_bit], axis=0).reshape(1, -1)
        # action = np.random.uniform(-5, 5, size=(self.action_dim))
        state = jax.device_put(state, device)
        action = self.modelstate.apply_fn({'params': self.modelstate.params}, state, rng,
                                          method="action")
        return np.array(action[0]).squeeze()
    
    def load(self, load_path):
        pass

class Actor(nn.Module):
    act_dims: int
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.silu(nn.Dense(256)(x))
        x = nn.LayerNorm()(x)
        x = nn.silu(nn.Dense(256)(x))+x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.act_dims*2)(x)
        # mu and log_std
        mu,log_std = jnp.split(x,2, axis=-1)
        # before_tanh = mu
        # jax.debug.print("DEBUG: before_tanh: {v}",v=jnp.max(before_tanh))
        mu = jnp.tanh(mu)
        # mu,log_std = jnp.split(x.reshape(-1, self.act_dims, 2),2, axis=-1)
        return mu,log_std
  
class Critic(nn.Module):
    @nn.compact
    def __call__(self, x, a):
        x = x.reshape((x.shape[0], -1))
        a = a.reshape((a.shape[0], -1))
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.silu(nn.Dense(256)(x))
        x = nn.LayerNorm()(x)
        x = nn.silu(nn.Dense(256)(x))+x
        x = nn.LayerNorm()(x)
        # x = nn.silu(nn.Dense(256)(x))+x
        # x = nn.LayerNorm()(x)
        x = nn.Dense(1)(x)
        return x
    
class SAC(nn.Module):
    obs_dims: int
    act_dims: int
    gamma: float = 0.999
    min_q_weight: float = 0.01
    num_act_samples: int = 20
    alpha_init: float = 0.05
    lambda_critics: float = 1.0
    entropy_target: float = 15
    
    def setup(self):
        self.actor = Actor(act_dims=self.act_dims)
        self.critic1 = Critic()
        self.critic2 = Critic()
        self.target_critic1 = Critic()
        self.target_critic2 = Critic()
        self.log_alpha = self.param("log_alpha", lambda key: jnp.log(self.alpha_init))
        
    def __call__(self, x, rng):
        # Only used for initialization
        
        mu,log_std = self.actor(x)
        act = self.reparameterize(mu, log_std, rng)
        q1 = self.critic1(x, act)
        q2 = self.critic2(x, act)
        # q = jnp.minimum(q1, q2)
        q = (q1+ q2)/2
        target_q1 = self.target_critic1(x, act)
        target_q2 = self.target_critic2(x, act)
        target_q = (target_q1+ target_q2)/2
        # target_q = jnp.minimum(target_q1, target_q2)
        return q, target_q, mu, log_std
    
    def action(self, x, rng):
        mu,log_std = self.actor(x)
        act = self.reparameterize(mu, log_std, rng)
        return act
    
    def reparameterize(self, mu, log_std, rng):
        std = jnp.exp(log_std)
        return mu + std * random.normal(rng, mu.shape)

    def log_pi(self, mu, log_std, act):
        std = jnp.exp(log_std)
        return jnp.sum(-0.5 * jnp.log(2 * jnp.pi) - log_std - 0.5 * jnp.square((act - mu) / std), axis=-1)

    def actor_loss(self, x, rng):
        mu,log_std = self.actor(x)
        log_std = jnp.clip(log_std, None, 30)
        act = self.reparameterize(mu, log_std, rng)
        
        alpha = jnp.exp(self.log_alpha)
        
        q1 = self.critic1.apply({
            'params': jax.lax.stop_gradient(self.critic1.variables['params'])}, 
                               x, act)
        q2 = self.critic2.apply({
            'params': jax.lax.stop_gradient(self.critic2.variables['params'])}, 
                               x, act)
        q = (q1+ q2)/2
        # q = jnp.minimum(q1, q2)
        
        log_pi = self.log_pi(mu, log_std, act)
        qmean = q.mean()
        entropy = -log_pi.mean()
        alpha_loss = -alpha * jax.lax.stop_gradient(entropy - self.entropy_target)
        # return (-q + log_pi* self.alpha).mean()
        return -qmean-entropy*jax.lax.stop_gradient(alpha), alpha_loss, \
                    {"q_actor":qmean, 
                     "entropy":entropy, 
                     "log_std_mean":jnp.mean(log_std), 
                     "log_std_max":jnp.max(log_std),
                     "alpha":alpha
                     }
    
    def critic_loss(self, s, a, r, s_prime, a_prime,done, rng):
        rng1, rng2 = random.split(rng)
        
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        
        q1_target = self.target_critic1(s_prime, a_prime)
        q2_target = self.target_critic2(s_prime, a_prime)
        q_target = (q1_target+ q2_target)/2
        # q_target = jnp.minimum(q1_target, q2_target)
        td_target = jax.lax.stop_gradient(r + self.gamma * q_target*(1-done))
        
        td1_loss = optax.l2_loss(q1, (td_target)).mean()
        td2_loss = optax.l2_loss(q2, (td_target)).mean()
        td_loss = (td1_loss + td2_loss)/2.0
        
        # mu,logstd = self.actor(s)
        # mu,logstd = jax.lax.stop_gradient(mu), jax.lax.stop_gradient(logstd)
        # # sample action from current policy and uniform random
        # replicate_mu = jnp.repeat(mu, self.num_act_samples, axis=0)
        # replicate_logstd = jnp.repeat(logstd, self.num_act_samples, axis=0)
        
        # shape: (batch_size*num_act_samples, act_dim)
        # dataset_act = self.reparameterize(replicate_mu, replicate_logstd, rng1)
        dataset_act = a.repeat(self.num_act_samples, axis=0)
        random_act = random.uniform(rng2, shape=dataset_act.shape, minval=-1, maxval=1)
        
        replicate_s = jnp.repeat(s, self.num_act_samples, axis=0)
        q1_dataset = self.critic1(replicate_s, dataset_act)
        q2_dataset = self.critic2(replicate_s, dataset_act)
        q_dataset = (q1_dataset+ q2_dataset)/2
        q_dataset_mean = q_dataset.mean()
        
        q1_random = self.critic1(replicate_s, random_act)
        q2_random = self.critic2(replicate_s, random_act)
        q_random = (q1_random+ q2_random)/2
        q_random_mean = q_random.mean()
        
        # q_dataset = jnp.minimum(q1_dataset, q2_dataset)
        # q_random = jnp.minimum(q1_random, q2_random)
        
        min_q_loss = q_random_mean - q_dataset_mean
        
        return td_loss + min_q_loss*self.min_q_weight, \
                    {"td_loss":td_loss, 
                     "min_q_loss":min_q_loss, 
                     "q_target":q_target.mean(),
                    "q_dataset":q_dataset_mean, 
                    "q_random":q_random_mean
                    }
                    
    
    def compute_loss(self, s, a, r, s_prime, a_prime, done, rng):
        rng1, rng2 = random.split(rng)
        actor_loss, alpha_loss, actor_aux = self.actor_loss(s, rng1)
        critic_loss, critic_aux= self.critic_loss(s, a, r, s_prime, a_prime,done, rng2)
        total_loss = actor_loss+ alpha_loss + critic_loss*self.lambda_critics
        # total_loss = actor_loss + critic_loss*self.lambda_critics
        
        return total_loss, {
            "actor_loss":actor_loss, 
            "critic_loss":critic_loss, 
            "alpha_loss":alpha_loss,
            **actor_aux,
            **critic_aux
            }
        # return actor_loss + critic_loss

@struct.dataclass
class Metrics(metrics.Collection):
    # accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss') 
    actor_loss: metrics.Average.from_output('actor_loss') 
    critic_loss: metrics.Average.from_output('critic_loss')
    alpha_loss: metrics.Average.from_output('alpha_loss')
    
    alpha: metrics.Average.from_output('alpha')
    q_actor: metrics.Average.from_output('q_actor')
    q_target: metrics.Average.from_output('q_target')
    q_dataset: metrics.Average.from_output('q_dataset')
    q_random: metrics.Average.from_output('q_random')
    
    entropy: metrics.Average.from_output('entropy')
    td_loss: metrics.Average.from_output('td_loss')
    min_q_loss: metrics.Average.from_output('min_q_loss')
    grads_norm: metrics.Average.from_output('grads_norm')
    log_std_mean: metrics.Average.from_output('log_std_mean')
    log_std_max: metrics.Average.from_output('log_std_max')
        
    
class TrainState(train_state.TrainState):
    
    metrics: Metrics
    
def create_train_state(model:nn.Module ,rng:jax.Array, learning_rate, beta1):
    """Creates an initial `TrainState`."""
    params = model.init(rng, jnp.ones([1, OBS_DIM+1]),rng)['params'] # initialize parameters by passing a template image
    # tx = optax.sgd(learning_rate, beta1)
    # tx = optax.adam(learning_rate, beta1)
    tx = optax.chain(
        optax.clip_by_global_norm(100.0),
        # optax.add_decayed_weights(1e-5),
        optax.adam(learning_rate, beta1)
    )
    # tx = optax.sgd(learning_rate, momentum)
    # state= TrainState.create(
    #     apply_fn=model.apply, params=FrozenDict(params), tx=tx,
    #     metrics=Metrics.empty())
    state= TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
        metrics=Metrics.empty())
    return state

def soft_update_target(state:TrainState,tau=0.9):
    
    # new_tp1= jax.tree.map(lambda p,tp: p*tau + tp*(1-tau),state.params['critic1'] ,state.params['target_critic1'] )
    # new_tp2= jax.tree.map(lambda p,tp: p*tau + tp*(1-tau),state.params['critic2'] ,state.params['target_critic2'] )
    state.params['target_critic1']= jax.tree.map(lambda p,tp: p*tau + tp*(1-tau),state.params['critic1'] ,state.params['target_critic1'] )
    state.params['target_critic2']= jax.tree.map(lambda p,tp: p*tau + tp*(1-tau),state.params['critic2'] ,state.params['target_critic2'] )
    # new_params = state.params.copy({"target_critic1":new_tp1,
                                        # "target_critic2":new_tp2})
    # return state.replace(params=new_params)
    # return state
    ...

@jax.jit
def train_step(state, batch,rng):
    """Train for a single step."""
    def loss_fn(params):
        # pred = state.apply_fn({'params': params}, batch['obs'])
        # loss = optax.l2_loss(pred, batch['act']).mean()
        total_loss, aux = state.apply_fn({'params':params},batch['obs'], batch['act'], batch['rew'], batch['obs_prime'], batch['act_prime'], batch['dones'], rng,method='compute_loss')
        return total_loss, aux
    
    (loss, aux),grads = jax.value_and_grad(loss_fn,has_aux=True)(state.params)
    grads_norm = optax.global_norm(grads)
    # jax.debug.print("Total loss {loss}",loss=loss)
    state = state.apply_gradients(grads=grads)
    state = state.replace(metrics=state.metrics.merge(
        Metrics.single_from_model_output(
            loss=loss,
            grads_norm = grads_norm,
            **aux
                                         )))
    return state

# def debug_setup():
#     os.environ['JAX_DEBUG_NANS'] = "True"
    ...

def train(exp_name="",SAMPLE_EXAMPLE=False,VERBOSE=1, USE_WANDB=True):
    # VERBOSE: 0,1,2
        # 0: No print
        # 1: Print loss after each epoch
    if USE_WANDB:
        os.environ['WANDB_SILENT'] = 'true'
        import wandb
        wandb.init(project="rlp", name="cqlsac_"+exp_name+"_"+time.strftime("%m%d-%H:%M:%S") , save_code=True)
    np.random.seed(SEED)
    init_rng = jax.random.key(SEED)
    init_rng, rng1, rng2, rng3 = random.split(init_rng, 4)

    num_epochs = 500
    learning_rate = 0.0003
    beta1 = 0.99
    batch_size = 512
    test_size = 0.2
    target_update_interval=5
    tau = 0.9
    
    data = make_dataset(MAKE_SARSA=True)
    merge_data = merge_dataset(data['walk_mr'])
    merge_data, rew_mean, rew_std = reward_normalize(merge_data)
    print("Reward Normalize Mean, Std: ", rew_mean, rew_std)
    if USE_WANDB:
        wandb.config.update({"reward_mean":rew_mean, "reward_std":rew_std})
        # wandb.config.reward_mean = rew_mean
        # wandb.config.reward_std = rew_std
    train_ds = DataLoader(merge_data, batch_size=batch_size)
    print("Train Dataloader Size",len(train_ds))
    
    # learning_rate = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=num_epochs*len(train_ds), alpha=0.0)
    
    model = SAC(obs_dims=OBS_DIM, act_dims=ACT_DIM)
    state: TrainState = create_train_state(model, rng2, learning_rate, beta1)
    print(model.tabulate(rng3, jnp.ones([1, OBS_DIM+1]),rng3,compute_flops=True))
    state = jax.device_put(state, device)
    
    
    for epoch in range(num_epochs):
        
        for step,batch in enumerate(train_ds):
            rng1, rng2 = random.split(rng1, 2)
            state = train_step(state, batch, rng2)
            if step%target_update_interval==0:
                # import pdb;pdb.set_trace()
                soft_update_target(state,tau)
        metric_res = state.metrics.compute()
        
        state = state.replace(metrics=state.metrics.empty()) 
        
        if USE_WANDB:
            wandb.log(metric_res, step=epoch)
        if VERBOSE>=1:
            print(f"train epoch\t: {epoch}, \n",
                  *[f"\t{key}\t: {value}, \n" for key,value in metric_res.items()]
                  )
            
    checkpoints.save_checkpoint(ckpt_dir=os.path.join(PWD, 'ckpt','cql'),
                            target=state,
                            step=0,
                            overwrite=True,
                            keep=2)
    
    if SAMPLE_EXAMPLE:
        # sample a batch of data
        batch = next(train_ds)
        # inference
        num_sample = 10 
        rng1, rng2 = random.split(rng1)
        act = state.apply_fn({'params': state.params}, batch['obs'][:num_sample],rng2, method="action")
        loss = optax.l2_loss(act, batch['act'][:num_sample]).mean()
        print("DEBUG: batch['obs']", batch['obs'][:num_sample])
        print("DEBUG: act", act)
        print("DEBUG: batch['act']", batch['act'][:num_sample])
        print("DEBUG: difference", loss)
    
            
    ...
    
def test():
    np.random.seed(SEED)
    init_rng = jax.random.key(SEED)
    rng1, rng2, rng3 = random.split(init_rng, 3)

    learning_rate = 0.005
    momentum = 0.9
    
    model = SAC(obs_dims=OBS_DIM, act_dims=ACT_DIM, gamma=0.99, min_q_weight=0.5, num_act_samples=10)
    state = create_train_state(model, rng2, learning_rate, momentum)
    state = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(PWD, 'ckpt','cql'), target=state)
    state = jax.device_put(state, device)
    
    agent = CQLSACAgent(state, OBS_DIM, ACT_DIM, rng=rng3)
    eval_agent(agent, eval_episodes=5,seed=SEED)
    

if __name__=="__main__":
    try:
        start_time = time.time()
        fire.Fire({
            'train': train,
            'test': test
        })
        end_time = time.time()
        print("Program Running time: ", end_time-start_time)
    except Exception as e:
        print(">>BUG: ",e)
        import pdb;pdb.post_mortem()