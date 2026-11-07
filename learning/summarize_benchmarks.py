"""Aggregate every trial, including failures; never declare convergence."""
import argparse
import json
from pathlib import Path
import statistics
import copy


def moments(values):
    values=[float(v) for v in values if v is not None]
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
                sd=statistics.stdev(values) if len(values)>1 else None)


def summarize(reports):
    groups=[]
    fields=['return_sum','area_gain_m2','sim_seconds','navigation_success_rate','area_per_sim_minute']
    for report in reports:
        episodes=copy.deepcopy(report['episodes'])
        for e in episodes:
            steps=e.get('steps',[])
            e['navigation_success_rate']=(sum(s.get('result')=='succeeded' for s in steps)/len(steps) if steps else 0.)
            e['duration_complete']=all(s.get('valid_transition') for s in steps)
            if not e['duration_complete']:
                e['area_per_sim_minute']=None
        groups.append(dict(label=report['label'],policy=report['policy'],completed=report['completed'],
            checkpoint_sha256=report.get('checkpoint_sha256'),protocol=report['protocol'],
            trials=len(episodes),errors=sum(bool(e.get('error') or e.get('shutdown_error')) for e in episodes),
            interrupted=sum(e['stop_reason'] not in ('step_limit','decision_budget','simulation_budget_at_action_boundary','no_valid_frontier') for e in episodes),
            metrics={field:moments(e.get(field) for e in episodes) for field in fields},episodes=episodes))
    paired=[]
    trained=next((g for g in groups if g['label']=='trained'),None)
    if trained:
        for other in groups:
            if other is trained or other['protocol']!=trained['protocol']:
                continue
            left={e['seed']:e for e in trained['episodes']}
            right={e['seed']:e for e in other['episodes']}
            seeds=sorted(left.keys() & right.keys())
            paired.append(dict(reference=other['label'],seeds=seeds,
                return_difference=moments(left[s]['return_sum']-right[s]['return_sum'] for s in seeds),
                area_difference=moments(left[s]['area_gain_m2']-right[s]['area_gain_m2'] for s in seeds)))
    return dict(groups=groups,paired_differences=paired,convergence_claim=False,
        notes=['Mean and sample standard deviation, not confidence intervals.',
               'Same world/start and seeds; Gazebo/SLAM are not bitwise deterministic.',
               'Failures retained with achieved partial return/area; interruptions shown separately.',
               'Navigation success denominator includes all attempted actions, including invalidated ones.',
               'Area/time efficiency omitted when an interrupted action has no complete duration.',
               'Time cap checked at action boundaries; actual simulated durations and overshoot retained.',
               'Local free-area proxy is not ground-truth house coverage.'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    reports=[json.loads(f.read_text()) for f in sorted(args.root.rglob('benchmark.json'))]
    result=summarize(reports)
    path=args.root/'comparison.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False))
    temp.replace(path)
    print(json.dumps({g['label']:g['metrics'] for g in result['groups']}))


if __name__=='__main__':
    main()
