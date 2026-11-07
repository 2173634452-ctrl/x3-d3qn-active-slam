"""Synthetic global-revision injection during real motion, followed by cold reset."""
import argparse,json,traceback
from pathlib import Path
from gym_env import X3ExplorationEnv

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
env=X3ExplorationEnv(a.output,enable_fault_injection=True)
report=dict(passed=False,synthetic_revision=True,natural_loop_closure_tested=False)
try:
    obs,info=env.reset(seed=42)
    report['initial']=info
    obs,reward,terminated,truncated,info=env.inject_map_revision_for_test(moving=True)
    report['fault']=dict(terminated=terminated,truncated=truncated,info=info)
    fault_ok=(truncated and not terminated and not info['valid_transition'] and
              info['navigation_result'] in ('canceled','failed') and info['known_cells']==0 and
              sum(map(abs,info['command'])) < 1e-7)
    obs,info=env.reset(seed=42)
    report['reset']=info
    report['passed']=bool(fault_ok and info['step']==0 and any(obs['action_mask']) and info['epoch']==[0,0])
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
    (a.output/'moving_revision_acceptance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if not report['passed']:
    raise SystemExit(1)
