"""Autonomous behaviour-policy collection; no neural network or optimizer."""
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exploration'))
from gym_env import X3ExplorationEnv

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--actor',type=int,required=True)
p.add_argument('--episodes',type=int,default=2)
p.add_argument('--steps',type=int,default=8)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
env=X3ExplorationEnv(a.output/'simulation',max_steps=a.steps)
rng=np.random.default_rng(1000+a.actor)
dataset=open(a.output/'transitions.jsonl','w',buffering=1)
report=dict(actor=a.actor,phase='collection',network_updates=0,samples=0,attempts=0,episodes=[])
started=time.time()
try:
    for episode in range(a.episodes):
        obs,info=env.reset(seed=1000+a.actor*100+episode)
        rec=dict(session=info['session'],results=[])
        report['episodes'].append(rec)
        for step in range(a.steps):
            valid=np.flatnonzero(obs['action_mask'])
            if not len(valid):
                break
            best=max(map(int,valid),key=lambda i:info['candidates'][i]['score']/(1+info['candidates'][i]['visits']))
            action=int(rng.choice(valid)) if rng.random()<.35 else best
            before=info
            nxt,reward,terminated,truncated,info=env.step(action)
            report['attempts']+=1
            rec['results'].append(info.get('result',info['reason']))
            if info.get('valid_transition',False):
                row=dict(episode_id=before.get('session',rec['session']),actor=a.actor,step=step,
                    state=obs['vector'].tolist(),action_mask=obs['action_mask'].tolist(),
                    candidates=before['candidates'],snapshot_id=before['snapshot_id'],action=action,
                    behavior='frontier_score_epsilon_0.35',behavior_probability=.35/len(valid)+(.65 if action==best else 0),
                    reward=reward,reward_terms=info['reward_terms'],next_state=nxt['vector'].tolist(),
                    next_action_mask=nxt['action_mask'].tolist(),next_candidates=info['candidates'],
                    terminated=terminated,truncated=truncated,duration=info['sim_duration'],
                    result=info.get('result'),epoch=before.get('epoch'),valid_transition=True)
                dataset.write(json.dumps(row)+'\n')
                report['samples']+=1
            obs=nxt
            tmp=a.output/'collector.tmp'
            tmp.write_text(json.dumps(report))
            tmp.replace(a.output/'collector_progress.json')
            print(json.dumps(dict(actor=a.actor,episode=episode,step=step,result=info.get('result'),samples=report['samples'])),flush=True)
            if terminated or truncated:
                break
except Exception as exc:
    report.update(error=str(exc),traceback=traceback.format_exc())
finally:
    try:
        env.close()
    except Exception as exc:
        report['shutdown_error']=str(exc)
    dataset.close()
    report.update(elapsed_seconds=time.time()-started,shutdowns=env.completed_sessions,finished_at=time.time())
    (a.output/'collection_report.json').write_text(json.dumps(report,indent=2))
