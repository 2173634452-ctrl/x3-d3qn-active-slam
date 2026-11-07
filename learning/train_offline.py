"""D3QN pretraining on a frozen, real transition dataset; no Gazebo calls."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from d3qn import Learner

p=argparse.ArgumentParser()
p.add_argument('--dataset',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--updates',type=int,default=1000)
p.add_argument('--resume',type=Path)
p.add_argument('--checkpoint-every',type=int,default=1000)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
data=a.dataset.read_bytes()
rows=[json.loads(s) for s in data.decode().splitlines() if s]
if len(rows)<8:
    raise RuntimeError('Need at least 8 real transitions for this pretraining run')
l=Learner(batch_size=min(32,len(rows)),lr=3e-4,target_interval=100)
initial_updates=0
if a.resume:
    saved=torch.load(a.resume,map_location='cpu',weights_only=True)
    l.online.load_state_dict(saved['policy'])
    l.target.load_state_dict(saved['policy'])
    initial_updates=int(saved.get('updates',0))
    l.updates=initial_updates
before={k:v.clone() for k,v in l.online.state_dict().items()}
for row in rows:
    obs=dict(vector=np.array(row['state'],np.float32),action_mask=np.array(row['action_mask'],np.int8))
    nxt=dict(vector=np.array(row['next_state'],np.float32),action_mask=np.array(row['next_action_mask'],np.int8))
    assert obs['action_mask'][row['action']]
    l.remember(obs,row['action'],row['reward'],nxt,row['terminated'],row['truncated'],
        dict(valid_transition=True,sim_duration=row['duration']))
started=time.time()
metrics=[]
def save_checkpoint(path):
    probe=torch.arange(36,dtype=torch.float32).reshape(1,36)/36.
    with torch.no_grad():
        q=l.online(probe).tolist()
    tmp=path.with_suffix('.tmp')
    torch.save(dict(policy=l.online.state_dict(),updates=l.updates,probe_q=q,
        observation_contract='x3-frontier-engineering-v1',training='fixed_dataset_d3qn_pretraining',
        trained_for_performance=False),tmp)
    tmp.replace(path)
for _ in range(a.updates):
    value=l.update()
    if value is None or not np.isfinite(value['loss']):
        raise RuntimeError('Invalid offline optimization step')
    metrics.append(value)
    if l.updates%50==0:
        tmp=a.output/'progress.tmp'
        tmp.write_text(json.dumps(dict(updates=l.updates,loss=value['loss'],samples=len(rows))))
        tmp.replace(a.output/'train_progress.json')
    if (l.updates-initial_updates)%a.checkpoint_every==0:
        save_checkpoint(a.output/'d3qn_latest.pt')
probe=torch.arange(36,dtype=torch.float32).reshape(1,36)/36.
with torch.no_grad():
    probe_q=l.online(probe).tolist()
checkpoint=a.output/'d3qn_offline.pt'
torch.save(dict(policy=l.online.state_dict(),updates=l.updates,probe_q=probe_q,
    observation_contract='x3-frontier-engineering-v1',training='fixed_dataset_d3qn_pretraining',
    trained_for_performance=False),checkpoint)
assert hashlib.sha256(a.dataset.read_bytes()).hexdigest()==hashlib.sha256(data).hexdigest()
report=dict(passed=True,method='Dueling Double DQN fixed-dataset pretraining',samples=len(rows),updates=l.updates,
    batch_size=l.batch_size,target_interval=100,learning_rate=3e-4,
    dataset_sha256=hashlib.sha256(data).hexdigest(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    elapsed_seconds=time.time()-started,device='cpu',metrics=metrics,
    parameter_delta_l2=sum(float((v-before[k]).square().sum()) for k,v in l.online.state_dict().items())**.5,
    collection_network_updates=0,training_simulator_interactions=0,
    convergence_claim=False)
report.update(initial_updates=initial_updates,new_updates=a.updates,
    resume=str(a.resume) if a.resume else None,resume_mode='policy_weights_with_fresh_optimizer')
(a.output/'offline_training_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='metrics'}),flush=True)
