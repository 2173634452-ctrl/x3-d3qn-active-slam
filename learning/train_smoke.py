"""Small online learning proof, followed by inference in a separate process."""
import argparse,hashlib,json,os,shutil,subprocess,sys,traceback
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exploration'))
from gym_env import X3ExplorationEnv
from d3qn import Learner

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--episodes',type=int,default=3)
p.add_argument('--steps',type=int,default=3)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
snapshot=a.output/'learning_source'
snapshot.mkdir()
for file in Path(__file__).resolve().parent.iterdir():
    if file.is_file() and (file.suffix in ('.py','.sh') or file.name=='Dockerfile'):
        shutil.copy2(file,snapshot/file.name)
(a.output/'learning_source_sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest()
    for f in snapshot.iterdir()},indent=2))
l=Learner()
initial={k:v.clone() for k,v in l.online.state_dict().items()}
env=X3ExplorationEnv(a.output/'training',max_steps=a.steps)
report=dict(passed=False,training_episodes=[],updates=[],network='36-128-64; V:64-32-1; A:64-32-10',
            device='cpu',performance_claim=False,torch_version=str(torch.__version__))
try:
    for episode in range(a.episodes):
        obs,info=env.reset(seed=42+episode)
        record=dict(session=info['session'],steps=[])
        report['training_episodes'].append(record)
        for step in range(a.steps):
            action=l.act(obs,epsilon=max(.2,.8-.2*episode))
            next_obs,reward,terminated,truncated,info=env.step(action)
            inserted=l.remember(obs,action,reward,next_obs,terminated,truncated,info)
            record['steps'].append(dict(action=action,reward=reward,result=info.get('result'),
                reason=info['reason'],inserted=inserted,terminated=terminated,truncated=truncated))
            for _ in range(2):
                metrics=l.update()
                if metrics is not None:
                    report['updates'].append(metrics)
            obs=next_obs
            print(json.dumps(dict(episode=episode+1,step=step+1,result=info.get('result'),
                replay=len(l.buffer),updates=l.updates)),flush=True)
            if terminated or truncated:
                break
    env.close()
    delta=sum(float((value-initial[k]).square().sum()) for k,value in l.online.state_dict().items())**.5
    checkpoint=a.output/'d3qn_smoke.pt'
    probe=torch.arange(36,dtype=torch.float32).reshape(1,36)/36.
    with torch.no_grad():
        probe_q=l.online(probe).tolist()
    torch.save(dict(policy=l.online.state_dict(),architecture=report['network'],
        observation_contract='x3-frontier-engineering-v1',updates=l.updates,probe_q=probe_q,
        trained_for_performance=False),checkpoint)
    report.update(parameter_delta_l2=delta,replay_size=len(l.buffer),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_checkpoint.py')),
        '--checkpoint',str(checkpoint),'--output',str(a.output/'evaluation')],check=True)
    evaluation=json.loads((a.output/'evaluation/inference_acceptance.json').read_text())
    report['evaluation']=evaluation
    report['passed']=bool(l.updates>=2 and delta>0 and evaluation['passed'] and
        all(m['value_grad_norm']>0 and m['advantage_grad_norm']>0 and np.isfinite(m['loss']) for m in report['updates']))
except Exception as exc:
    report['error']=str(exc)
    report['traceback']=traceback.format_exc()
finally:
    try:
        env.close()
    except Exception as exc:
        report['shutdown_error']=str(exc)
        report['passed']=False
    report['training_shutdowns']=env.completed_sessions
    (a.output/'training_acceptance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if not report['passed']:
    raise SystemExit(1)
