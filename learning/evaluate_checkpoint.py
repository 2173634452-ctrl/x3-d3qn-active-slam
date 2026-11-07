"""Fresh-process greedy inference: load exactly once; perform no updates."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exploration'))
from gym_env import X3ExplorationEnv
from d3qn import DuelingQ

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
net=DuelingQ()
net.load_state_dict(saved['policy'])
net.eval()
with torch.no_grad():
    actual=net(torch.arange(36,dtype=torch.float32).reshape(1,36)/36.).numpy()
error=float(np.max(np.abs(actual-np.array(saved['probe_q']))))
report=dict(passed=False,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            load_probe_max_error=error,optimizer_updates=0,steps=[])
env=X3ExplorationEnv(a.output/'simulation',max_steps=2)
try:
    obs,info=env.reset(seed=99)
    for _ in range(2):
        valid=np.flatnonzero(obs['action_mask'])
        with torch.no_grad():
            q=net(torch.tensor(obs['vector'][None,:],dtype=torch.float32))[0].numpy()
        action=int(valid[np.argmax(q[valid])])
        obs,reward,terminated,truncated,info=env.step(action)
        report['steps'].append(dict(action=action,valid_actions=valid.tolist(),q_values=q.tolist(),
            reward=reward,result=info.get('result'),reason=info['reason']))
        if terminated or truncated:
            break
    report['passed']=bool(error==0. and any(s['result']=='succeeded' for s in report['steps']))
except Exception as exc:
    report['error']=str(exc)
    report['traceback']=traceback.format_exc()
finally:
    try:
        env.close()
    except Exception as exc:
        report['shutdown_error']=str(exc)
        report['passed']=False
    report['shutdowns']=env.completed_sessions
    (a.output/'inference_acceptance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if not report['passed']:
    raise SystemExit(1)
