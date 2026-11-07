"""Bounded collection -> fixed-dataset D3QN -> evaluation, in one container.

Each collector has its own output directory and Gazebo partition. Model updates
start only after every collector exits. No viewer or recording is started.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(module, *args, env=None):
    subprocess.run([sys.executable, str(ROOT / 'learning' / module), *map(str, args)], check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--actors', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--samples-per-actor', type=int, default=250)
    parser.add_argument('--max-hours', type=float, default=8)
    parser.add_argument('--updates', type=int, default=10000)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.samples_per_actor < 8 or args.updates < 1 or args.max_hours <= 0:
        parser.error('Require >=8 samples per actor, positive updates and hours')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['X3_SAVE_VIDEO'] = '0'
    os.environ['X3_RECORD_VIDEO'] = '0'
    state = dict(started=time.time(), phase='collection', completed=False)

    def status():
        temp = args.output / 'pipeline.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(args.output / 'pipeline.json')

    status()
    try:
        def collect(actor):
            run('collect_long.py', '--output', args.output / f'actor_{actor}',
                '--actor', actor, '--target-samples', args.samples_per_actor,
                '--max-hours', args.max_hours, '--steps', 20,
                env=dict(os.environ, X3_ROS_DOMAIN_BASE=str(20 + actor * 20)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.actors) as pool:
            list(pool.map(collect, range(args.actors)))
        dataset = args.output / 'transitions.jsonl'
        count = 0
        with dataset.open('w') as dest:
            for actor in range(args.actors):
                for line in (args.output / f'actor_{actor}' / 'transitions.jsonl').read_text().splitlines():
                    if line.strip():
                        json.loads(line)
                        dest.write(line + '\n')
                        count += 1
        if count < 8:
            raise RuntimeError(f'Only {count} transitions collected; refusing training')
        state.update(phase='training', samples=count)
        status()
        resume = ['--resume', args.resume] if args.resume else []
        run('train_offline.py', '--dataset', dataset, '--output', args.output / 'offline',
            '--updates', args.updates, *resume)
        state['phase'] = 'evaluation'
        status()
        run('evaluate_checkpoint.py', '--checkpoint', args.output / 'offline/d3qn_offline.pt',
            '--output', args.output / 'evaluation')
        state.update(phase='finished', completed=True)
    except BaseException as error:
        state.update(phase='failed', error=str(error))
        raise
    finally:
        state['finished'] = time.time()
        status()


if __name__ == '__main__':
    main()
