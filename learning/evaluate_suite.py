"""Run baselines and named checkpoints with isolated ROS domains."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    p.add_argument('--decisions',type=int,default=6)
    p.add_argument('--sim-budget',type=float,default=180)
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=4)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    jobs=[('random','random'),('heuristic','heuristic'),('untrained','untrained'),('checkpoint','trained')]

    def run(job):
        index,(policy,label)=job
        cmd=[sys.executable,str(root/'benchmark.py'),'--policy',policy,'--label',label,
             '--output',str(args.output/label),'--seeds',*map(str,args.seeds),
             '--decisions',str(args.decisions),'--sim-budget',str(args.sim_budget)]
        if policy=='checkpoint':
            cmd+=['--checkpoint',str(args.checkpoint)]
        with (args.output/(label+'.log')).open('w') as log:
            result=subprocess.run(cmd,env=dict(os.environ,X3_ROS_DOMAIN_BASE=str(20+index*40)),
                stdout=log,stderr=subprocess.STDOUT)
        return dict(label=label,exit_code=result.returncode)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(run,enumerate(jobs)))
    (args.output/'suite.json').write_text(json.dumps(dict(jobs=results,completed=all(r['exit_code']==0 for r in results)),indent=2))
    subprocess.run([sys.executable,str(root/'summarize_benchmarks.py'),'--root',str(args.output)],check=True)
    if any(r['exit_code'] for r in results):
        raise SystemExit('Evaluation worker failed; see suite.json and logs')


if __name__=='__main__':
    main()
