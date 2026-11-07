"""Bounded long collection with per-episode recovery and durable transitions."""
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exploration'))
from gym_env import X3ExplorationEnv

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--actor',type=int,required=True)
p.add_argument('--target-samples',type=int,default=250)
p.add_argument('--max-hours',type=float,default=8)
p.add_argument('--steps',type=int,default=20)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(7000+a.actor)
report=dict(actor=a.actor,phase='collection',network_updates=0,samples=0,attempts=0,
    target_samples=a.target_samples,episodes=[],errors=[],shutdowns=[])
started=time.time()
deadline=started+a.max_hours*3600
dataset=open(a.output/'transitions.jsonl','a',buffering=1)

def progress():
    tmp=a.output/'collector.tmp'
    tmp.write_text(json.dumps(report))
    tmp.replace(a.output/'collector_progress.json')

try:
    episode=0
    consecutive_errors=0
    while report['samples']<a.target_samples and time.time()<deadline:
        episode+=1
        env=X3ExplorationEnv(a.output/f'attempt_{episode:05d}',max_steps=a.steps)
        record=dict(attempt=episode,results=[])
        try:
            obs,info=env.reset(seed=7000+a.actor*10000+episode)
            record['session']=info['session']
            for step in range(a.steps):
                if time.time()>=deadline or report['samples']>=a.target_samples:
                    break
                valid=np.flatnonzero(obs['action_mask'])
                if not len(valid):
                    break
                best=max(map(int,valid),key=lambda i:info['candidates'][i]['score']/(1+info['candidates'][i]['visits']))
                action=int(rng.choice(valid)) if rng.random()<.35 else best
                before=info
                nxt,reward,terminated,truncated,info=env.step(action)
                report['attempts']+=1
                record['results'].append(info.get('result',info['reason']))
                if info.get('valid_transition',False):
                    row=dict(episode_id=str(env.session),actor=a.actor,step=step,
                        state=obs['vector'].tolist(),action_mask=obs['action_mask'].tolist(),
                        candidates=before['candidates'],snapshot_id=before['snapshot_id'],action=action,
                        behavior='frontier_score_epsilon_0.35',
                        behavior_probability=.35/len(valid)+(.65 if action==best else 0),
                        reward=reward,reward_terms=info['reward_terms'],next_state=nxt['vector'].tolist(),
                        next_action_mask=nxt['action_mask'].tolist(),next_candidates=info['candidates'],
                        terminated=terminated,truncated=truncated,duration=info['sim_duration'],
                        result=info.get('result'),epoch=before.get('epoch'),valid_transition=True)
                    dataset.write(json.dumps(row)+'\n')
                    report['samples']+=1
                obs=nxt
                progress()
                print(json.dumps(dict(actor=a.actor,episode=episode,step=step,samples=report['samples'],
                    target=a.target_samples,result=info.get('result'))),flush=True)
                if terminated or truncated:
                    break
            consecutive_errors=0
        except Exception as exc:
            consecutive_errors+=1
            report['errors'].append(dict(attempt=episode,error=str(exc),time=time.time()))
            record['error']=str(exc)
        finally:
            try:
                env.close()
            except Exception as exc:
                record['shutdown_error']=str(exc)
                if env.host is not None:
                    # Do not reset into an old, still-running ROS stack.
                    raise RuntimeError('Unstopped episode; exit this owned container') from exc
            report['shutdowns'].extend(env.completed_sessions)
            report['episodes'].append(record)
            progress()
        if consecutive_errors>=5:
            print('Repeated episode errors: waiting 30 s before retry',flush=True)
            time.sleep(30)
            consecutive_errors=0
except Exception as exc:
    report.update(error=str(exc),traceback=traceback.format_exc())
finally:
    dataset.close()
    report.update(elapsed_seconds=time.time()-started,finished_at=time.time(),
        stop_reason='sample_target' if report['samples']>=a.target_samples else
                    ('time_budget' if time.time()>=deadline else 'worker_error'))
    progress()
    (a.output/'collection_report.json').write_text(json.dumps(report,indent=2))
