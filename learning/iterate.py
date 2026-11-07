"""Two bounded rounds: train, evaluate snapshots, policy collection, retrain.

Runs inside an owned container; all children stop when that container stops.
The monitor generator is shut down in finally. No video or HTTP server.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--resume',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=10000)
    p.add_argument('--checkpoint-every',type=int,default=5000)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    p.add_argument('--decisions',type=int,default=6)
    p.add_argument('--actors',type=int,choices=[1,2,3,4],default=2)
    p.add_argument('--samples-per-actor',type=int,default=250)
    p.add_argument('--collection-hours',type=float,default=2)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(X3_RECORD_VIDEO='0',X3_SAVE_VIDEO='0')
    state=dict(phase='starting',completed=False,started_at=time.time(),rounds=[],
        config=dict(updates=a.updates,checkpoint_every=a.checkpoint_every,seeds=a.seeds,
            decisions=a.decisions,actors=a.actors,samples_per_actor=a.samples_per_actor,
            collection_hours=a.collection_hours),convergence_claim=False)

    def status(phase):
        state['phase']=phase
        temp=a.output/'iteration.tmp'
        temp.write_text(json.dumps(state,indent=2))
        temp.replace(a.output/'iteration.json')

    def run(script,*args,env=None):
        with (a.output/(script+'.log')).open('a') as log:
            subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],check=True,
                stdout=log,stderr=subprocess.STDOUT,env=env)

    monitor_log=(a.output/'monitor.log').open('w')
    monitor=subprocess.Popen([sys.executable,str(ROOT/'build_dashboard.py'),'--root',str(a.output),
        '--output',str(a.output/'monitor'),'--watch-seconds','5'],stdout=monitor_log,stderr=subprocess.STDOUT)
    status('round_1_training')
    checkpoint=a.resume
    dataset=a.dataset
    try:
        for number in [1,2]:
            out=a.output/f'round_{number}'
            status(f'round_{number}_training')
            run('train_offline.py','--dataset',dataset,'--output',out/'offline',
                '--updates',a.updates,'--checkpoint-every',a.checkpoint_every,'--resume',checkpoint)
            status(f'round_{number}_evaluation')
            # Gradient updates already finished; evaluate retained snapshots in order.
            snapshots=sorted((out/'offline/checkpoints').glob('*.pt'))
            scores=[]
            for snapshot in snapshots:
                label=f'round_{number}_{snapshot.stem}'
                run('benchmark.py','--policy','checkpoint','--checkpoint',snapshot,
                    '--label',label,'--output',out/'evaluations'/label,
                    '--seeds',*a.seeds,'--decisions',a.decisions)
                report=json.loads((out/'evaluations'/label/'benchmark.json').read_text())
                episodes=report['episodes']
                scores.append(dict(checkpoint=str(snapshot),label=label,
                    mean_return=sum(e['return_sum'] for e in episodes)/len(episodes),
                    interruptions=sum(e['stop_reason'] not in ('step_limit','decision_budget',
                        'simulation_budget_at_action_boundary','no_valid_frontier') for e in episodes),
                    errors=sum(bool(e.get('error') or e.get('shutdown_error')) for e in episodes)))
                run('summarize_benchmarks.py','--root',a.output)
            # Do not discard failures from scores. Prefer reliable checkpoints for deployment.
            reliable=[s for s in scores if not s['interruptions'] and not s['errors']]
            best=max(reliable or scores,key=lambda s:s['mean_return'])
            checkpoint=out/'selected_policy.pt'
            shutil.copy2(best['checkpoint'],checkpoint)
            selection=dict(**best,fault_free_selection=bool(reliable),
                selection_is_test_performance=False,convergence_claim=False)
            (out/'selection.json').write_text(json.dumps(selection,indent=2))
            state['rounds'].append(dict(round=number,selection=selection,scores=scores))
            if number==2:
                break
            status('model_collection')
            def collect(actor):
                with (a.output/f'collector_{actor}.log').open('w') as log:
                    subprocess.run([sys.executable,str(ROOT/'collect_long.py'),
                        '--output',str(a.output/'collection'/f'actor_{actor}'),'--actor',str(actor),
                        '--target-samples',str(a.samples_per_actor),'--max-hours',str(a.collection_hours),
                        '--steps','20','--checkpoint',str(checkpoint),'--epsilon','0.35'],
                        env=dict(os.environ,X3_ROS_DOMAIN_BASE=str(20+actor*40)),
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                collection=json.loads((a.output/'collection'/f'actor_{actor}'/'collection_report.json').read_text())
                if any(e.get('shutdown_error') for e in collection['episodes']):
                    raise RuntimeError('Collector could not stop its episode; abort owned container')
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.actors) as pool:
                list(pool.map(collect,range(a.actors)))
            # Exact duplicate transitions are not added twice; original replay is retained.
            old=dataset
            dataset=a.output/'merged_dataset.jsonl'
            seen=set()
            new_count=0
            with dataset.open('w') as dest:
                for source in [old,*sorted((a.output/'collection').glob('actor_*/transitions.jsonl'))]:
                    for line in source.read_text().splitlines():
                        if not line.strip():
                            continue
                        row=json.loads(line)
                        key=json.dumps(row,sort_keys=True)
                        if key not in seen:
                            seen.add(key)
                            dest.write(json.dumps(row)+'\n')
                            if source!=old:
                                new_count+=1
            state.update(new_samples=new_count,merged_samples=len(seen),
                dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
            if not new_count:
                raise RuntimeError('No new valid transitions collected; not repeating unchanged-data training')
        state['completed']=True
        state['selected_checkpoint']=str(checkpoint)
        status('finished')
    except BaseException as error:
        state['error']=str(error)
        status('failed')
        raise
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
        monitor_log.close()
        state['finished_at']=time.time()
        status(state['phase'])
        run('build_dashboard.py','--root',a.output,'--output',a.output/'monitor')


if __name__=='__main__':
    main()
