"""Numerical checks addressing the bugs found in the authors' training code."""
import json,numpy as np,torch
from d3qn import Learner,double_target

l=Learner()
result=double_target(torch.tensor([[0.,2.,9.]]),torch.tensor([[8.,3.,50.]]),
    torch.tensor([[True,True,False]]),torch.tensor([1.]),torch.tensor([False]),torch.tensor([10.]))
assert torch.allclose(result,torch.tensor([3.97])),result
terminal=double_target(torch.tensor([[0.,2.,9.]]),torch.tensor([[8.,3.,50.]]),
    torch.tensor([[True,True,False]]),torch.tensor([1.]),torch.tensor([True]),torch.tensor([10.]))
assert terminal.item()==1.
try:
    l.online(torch.zeros(4,1,36))
    raise AssertionError('Ambiguous extra state dimension was accepted')
except ValueError:
    pass
for i in range(6):
    x=np.zeros(36,np.float32)
    x[i]=.4
    obs=dict(vector=x,action_mask=np.ones(10,np.int8))
    nx=dict(vector=x+.01,action_mask=np.ones(10,np.int8))
    assert l.remember(obs,i,float(i+1),nx,False,False,dict(valid_transition=True,sim_duration=10.))
before={k:v.clone() for k,v in l.online.state_dict().items()}
metrics=l.update()
assert metrics['batch_shape']==[4,36]
assert metrics['value_grad_norm']>0 and metrics['advantage_grad_norm']>0,metrics
assert any(not torch.equal(v,before[k]) for k,v in l.online.state_dict().items())
assert all(p.grad is None for p in l.target.parameters())
bad=dict(vector=np.zeros(36,np.float32),action_mask=np.zeros(10,np.int8))
for _ in range(100):
    masked=dict(vector=np.zeros(36,np.float32),action_mask=np.array([0,1,0,0,0,0,0,0,0,0],np.int8))
    assert l.act(masked,epsilon=1.)==1
assert not l.remember(masked,1,0.,bad,False,True,dict(valid_transition=False))
print(json.dumps(dict(passed=True,double_dqn_target=float(result),metrics=metrics,
    checks=['masked Double DQN target','terminal target','rank guard','nonzero advantage gradient',
            'parameter update','no target gradients','masked exploration','invalid-map replay exclusion'])))
