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
p.add_argument('--eval-seeds',type=int,nargs='+',help='Opt in to Gazebo baseline/checkpoint evaluation; no training during evaluation')
p.add_argument('--eval-decisions',type=int,default=6)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
if a.updates<1 or a.checkpoint_every<1:
    p.error('updates and checkpoint-every must be positive')
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
optimization_seconds=0.
metric_file=open(a.output/'metrics.jsonl','w',buffering=1)
def save_checkpoint(path):
    probe=torch.arange(36,dtype=torch.float32).reshape(1,36)/36.
    with torch.no_grad():
        q=l.online(probe).tolist()
    tmp=path.with_suffix('.tmp')
    torch.save(dict(policy=l.online.state_dict(),updates=l.updates,probe_q=q,
        observation_contract='x3-frontier-engineering-v1',training='fixed_dataset_d3qn_pretraining',
        trained_for_performance=False),tmp)
    tmp.replace(path)

def evaluate(path,label):
    if not a.eval_seeds:
        return
    import subprocess,sys
    subprocess.run([sys.executable,str(Path(__file__).with_name('benchmark.py')),
        '--policy','checkpoint','--checkpoint',str(path),'--label',label,
        '--output',str(a.output/'evaluations'/label),'--decisions',str(a.eval_decisions),
        '--seeds',*map(str,a.eval_seeds)],check=True)

initial=a.output/'checkpoints'/f'update_{l.updates:08d}.pt'
initial.parent.mkdir(exist_ok=True)
save_checkpoint(initial)
evaluate(initial,f'update_{l.updates:08d}')
for _ in range(a.updates):
    update_started=time.monotonic()
    value=l.update()
    optimization_seconds+=time.monotonic()-update_started
    if value is None or not np.isfinite(value['loss']):
        raise RuntimeError('Invalid offline optimization step')
    metrics.append(value)
    metric_file.write(json.dumps(value)+'\n')
    if l.updates%50==0:
        tmp=a.output/'progress.tmp'
        tmp.write_text(json.dumps(dict(updates=l.updates,loss=value['loss'],samples=len(rows))))
        tmp.replace(a.output/'train_progress.json')
    if (l.updates-initial_updates)%a.checkpoint_every==0:
        save_checkpoint(a.output/'d3qn_latest.pt')
        snapshot=a.output/'checkpoints'/f'update_{l.updates:08d}.pt'
        save_checkpoint(snapshot)
        evaluate(snapshot,f'update_{l.updates:08d}')
metric_file.close()
if a.updates%a.checkpoint_every:
    snapshot=a.output/'checkpoints'/f'update_{l.updates:08d}.pt'
    save_checkpoint(snapshot)
    evaluate(snapshot,f'update_{l.updates:08d}')
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
    optimization_seconds=optimization_seconds,
    parameter_delta_l2=sum(float((v-before[k]).square().sum()) for k,v in l.online.state_dict().items())**.5,
    collection_network_updates=0,training_simulator_interactions=0,
    convergence_claim=False)
report.update(initial_updates=initial_updates,new_updates=a.updates,
    resume=str(a.resume) if a.resume else None,resume_mode='policy_weights_with_fresh_optimizer')
(a.output/'offline_training_report.json').write_text(json.dumps(report,indent=2))
if a.eval_seeds:
    import shutil
    from summarize_benchmarks import summarize
    result=summarize([json.loads(f.read_text()) for f in (a.output/'evaluations').rglob('benchmark.json')])
    (a.output/'evaluations/comparison.json').write_text(json.dumps(result,indent=2))
    eligible=[g for g in result['groups'] if g['completed'] and g['trials']>=3 and not g['errors'] and not g['interrupted']]
    best=max(eligible,key=lambda g:g['metrics']['return_sum']['mean']) if eligible else None
    selection=dict(best_label=best['label'] if best else None,selection_metric='validation mean episode return',
        selection_is_test_performance=False,convergence_claim=False,
        reason='Best among complete >=3-trial, fault-free validation checkpoints' if best else 'No checkpoint passed eligibility criteria')
    if best:
        shutil.copy2(a.output/'checkpoints'/(best['label']+'.pt'),a.output/'d3qn_best_validation.pt')
    (a.output/'best_checkpoint.json').write_text(json.dumps(selection,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='metrics'}),flush=True)
