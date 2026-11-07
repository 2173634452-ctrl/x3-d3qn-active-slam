"""Repeatable, bounded policy comparisons. No optimizer, no recordings.

Seeds control policy randomness; the simulator restart does not guarantee
deterministic rendering/SLAM. Use multiple trials, retain all failures.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exploration'))
from gym_env import X3ExplorationEnv
from d3qn import DuelingQ


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    temp.replace(path)


def choose(policy,obs,info,rng,net):
    valid=np.flatnonzero(obs['action_mask'])
    if not len(valid):
        raise ValueError('No legal action')
    if policy=='random':
        return int(rng.choice(valid))
    if policy=='heuristic':
        return max(map(int,valid),key=lambda i:info['candidates'][i]['score']/(1+info['candidates'][i]['visits']))
    with torch.no_grad():
        q=net(torch.as_tensor(obs['vector'][None,:],dtype=torch.float32))[0].numpy()
    return int(valid[np.argmax(q[valid])])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--policy',choices=['random','heuristic','untrained','checkpoint'],required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--label')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    p.add_argument('--decisions',type=int,default=6)
    p.add_argument('--sim-budget',type=float,default=180)
    p.add_argument('--wall-limit',type=int,default=900)
    args=p.parse_args()
    if args.decisions<1 or args.sim_budget<=0 or args.wall_limit<80:
        p.error('Require positive decisions/time and wall-limit >=80')
    if args.policy=='checkpoint' and not args.checkpoint:
        p.error('checkpoint policy requires --checkpoint')
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(X3_RECORD_VIDEO='0',X3_SAVE_VIDEO='0',
        X3_EPISODE_WALL_LIMIT=str(args.wall_limit),X3_EPISODE_SIM_LIMIT=str(args.sim_budget+120))
    torch.set_num_threads(1)
    torch.manual_seed(42)
    net=DuelingQ().eval()
    checksum=None
    if args.checkpoint:
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        net.load_state_dict(checkpoint['policy'])
        checksum=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report=dict(policy=args.policy,label=args.label or args.policy,checkpoint_sha256=checksum,
        protocol=dict(decisions=args.decisions,sim_budget=args.sim_budget,seeds=args.seeds,
            budget_boundary='finish current action then stop; report any overshoot',
            initial_pose='world default; fresh full process restart',network_seed=42,
            simulator_deterministic=False,coverage_is_local_proxy=True),
        completed=False,episodes=[],optimizer_updates=0)
    atomic(args.output/'benchmark.json',report)
    for trial,seed in enumerate(args.seeds):
        episode=dict(trial=trial,seed=seed,steps=[],return_sum=0.,successes=0,
            valid_actions=0,sim_seconds=0.,area_gain_m2=0.,stop_reason='not_started',error=None)
        episode['duration_complete']=True
        env=X3ExplorationEnv(args.output/f'trial_{trial:02d}',max_steps=args.decisions)
        started=time.monotonic()
        broken=False
        try:
            obs,info=env.reset(seed=seed)
            rng=np.random.default_rng(seed)
            initial_area=float(info['ever_free_m2'])
            episode.update(initial_area_m2=initial_area,initial_base=info['base'],initial_epoch=info['epoch'])
            t0=float(info['sim_time'])
            for step in range(args.decisions):
                if not np.any(obs['action_mask']):
                    episode['stop_reason']='no_valid_frontier'
                    break
                action=choose(args.policy,obs,info,rng,net)
                obs,reward,terminated,truncated,info=env.step(action)
                valid=bool(info.get('valid_transition',False))
                entry=dict(step=step+1,action=action,reward=reward,result=info.get('result'),
                    reason=info.get('reason'),valid_transition=valid,
                    duration=info.get('sim_duration'),free_area_m2=info.get('ever_free_m2'))
                episode['steps'].append(entry)
                episode['successes']+=int(info.get('result')=='succeeded')
                if valid:
                    episode['return_sum']+=float(reward)
                    episode['valid_actions']+=1
                    episode['sim_seconds']=float(info['sim_time'])-t0
                    episode['area_gain_m2']=max(0.,float(info['ever_free_m2'])-initial_area)
                else:
                    episode['duration_complete']=False
                atomic(args.output/'current_episode.json',episode)
                if terminated or truncated:
                    episode['stop_reason']=info.get('reason','episode_end')
                    break
                if episode['sim_seconds']>=args.sim_budget:
                    episode['stop_reason']='simulation_budget_at_action_boundary'
                    break
            else:
                episode['stop_reason']='decision_budget'
        except Exception as error:
            episode.update(error=str(error),traceback=traceback.format_exc(),stop_reason='runtime_error')
        finally:
            try:
                env.close()
            except Exception as error:
                episode['shutdown_error']=str(error)
                broken=True
            episode['wall_seconds']=time.monotonic()-started
            episode['shutdowns']=env.completed_sessions
            episode['budget_overshoot_sim_seconds']=max(0.,episode['sim_seconds']-args.sim_budget)
            episode['navigation_success_rate']=(episode['successes']/len(episode['steps'])
                if episode['steps'] else 0.)
            episode['area_per_sim_minute']=(episode['area_gain_m2']*60/episode['sim_seconds']
                if episode['sim_seconds']>0 and episode['duration_complete'] else None)
            report['episodes'].append(episode)
            atomic(args.output/'benchmark.json',report)
        if broken:
            raise RuntimeError('Unverified process shutdown; abort owned container')
    report['completed']=True
    atomic(args.output/'benchmark.json',report)


if __name__=='__main__':
    main()
