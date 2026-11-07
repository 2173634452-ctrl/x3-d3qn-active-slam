"""Masked Dueling Double DQN for the 36-D engineering observation contract."""
from collections import deque
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DuelingQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(36,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.value=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        self.advantage=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,10))

    def forward(self,x):
        if x.ndim!=2 or x.shape[-1]!=36:
            raise ValueError('Expected a [batch,36] state tensor')
        h=self.features(x)
        advantage=self.advantage(h)
        return self.value(h)+advantage-advantage.mean(dim=-1,keepdim=True)


@torch.no_grad()
def double_target(online_next,target_next,next_mask,reward,terminated,duration,gamma=.99):
    available=next_mask.any(dim=1)
    selected=online_next.masked_fill(~next_mask,-torch.inf).argmax(dim=1)
    values=target_next.gather(1,selected[:,None]).squeeze(1)
    bootstrap=available & ~terminated
    # Macro actions take different amounts of simulation time. gamma is per 10 s.
    return reward+torch.pow(torch.as_tensor(gamma),duration/10.)*values*bootstrap.float()


class Learner:
    def __init__(self,seed=42,lr=1e-3,batch_size=4,target_interval=4):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(2)
        self.online=DuelingQ()
        self.target=DuelingQ()
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=lr)
        self.buffer=deque(maxlen=10000)
        self.batch_size=batch_size
        self.updates=0
        self.target_interval=target_interval

    @torch.no_grad()
    def act(self,observation,epsilon=0.):
        valid=np.flatnonzero(observation['action_mask'])
        if not len(valid):
            raise ValueError('No valid action')
        if random.random()<epsilon:
            return int(random.choice(valid.tolist()))
        q=self.online(torch.as_tensor(observation['vector'][None,:],dtype=torch.float32))[0].numpy()
        return int(valid[np.argmax(q[valid])])

    def remember(self,obs,action,reward,next_obs,terminated,truncated,info):
        if not info.get('valid_transition',False):
            return False
        if not obs['action_mask'][action]:
            raise ValueError('A masked action cannot enter the learning replay')
        self.buffer.append((obs['vector'].copy(),int(action),float(reward),
            next_obs['vector'].copy(),next_obs['action_mask'].copy(),bool(terminated),
            float(info['sim_duration']),bool(truncated)))
        return True

    def update(self):
        if len(self.buffer)<self.batch_size:
            return None
        rows=random.sample(list(self.buffer),self.batch_size)
        states,actions,rewards,next_states,masks,terminated,durations,_=zip(*rows)
        x=torch.tensor(np.stack(states),dtype=torch.float32)
        nx=torch.tensor(np.stack(next_states),dtype=torch.float32)
        action=torch.tensor(actions,dtype=torch.long)
        with torch.no_grad():
            target=double_target(self.online(nx),self.target(nx),torch.tensor(np.stack(masks),dtype=torch.bool),
                torch.tensor(rewards,dtype=torch.float32),torch.tensor(terminated,dtype=torch.bool),
                torch.tensor(durations,dtype=torch.float32))
        prediction=self.online(x).gather(1,action[:,None]).squeeze(1)
        loss=F.smooth_l1_loss(prediction,target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad=lambda module:sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)**.5
        value_grad,advantage_grad=grad(self.online.value),grad(self.online.advantage)
        nn.utils.clip_grad_norm_(self.online.parameters(),5.)
        self.optimizer.step()
        self.updates+=1
        if self.updates%self.target_interval==0:
            self.target.load_state_dict(self.online.state_dict())
        td=(prediction.detach()-target).abs()
        return dict(update=self.updates,loss=float(loss.detach()),td_abs_mean=float(td.mean()),
            q_mean=float(prediction.detach().mean()),q_abs_max=float(prediction.detach().abs().max()),
            target_mean=float(target.mean()),value_grad_norm=value_grad,
            advantage_grad_norm=advantage_grad,batch_shape=list(x.shape),
            target_requires_grad=target.requires_grad)
