"""One continuous, bounded exploration using the saved policy, without updates."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exploration'))
from gym_env import X3ExplorationEnv
from d3qn import DuelingQ

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--minutes',type=float,default=20)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
view=Path(os.environ['X3_VIEW_ROOT'])
os.environ['X3_EPISODE_WALL_LIMIT']=str(int(a.minutes*60+180))
os.environ['X3_EPISODE_SIM_LIMIT']=str(int(a.minutes*120))
torch.set_num_threads(2)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
net=DuelingQ();net.load_state_dict(saved['policy']);net.eval()
env=X3ExplorationEnv(a.output/'simulation',max_steps=100)
report=dict(completed=False,optimizer_updates=0,steps=[],max_minutes=a.minutes,
    checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
    complete_house_claim=False)
def status(message,detail):
    tmp=view/'status.tmp'
    tmp.write_text(json.dumps(dict(phase='evaluation',message=message,detail=detail),ensure_ascii=False))
    tmp.replace(view/'status.json')
try:
    obs,info=env.reset(seed=99)
    report['initial']=info
    started=time.monotonic()
    stagnant=failures=0
    for step in range(100):
        if time.monotonic()-started>=a.minutes*60:
            report['stop_reason']='wall_time_budget';break
        if (a.output/'STOP').exists():
            report['stop_reason']='user_stop';break
        valid=np.flatnonzero(obs['action_mask'])
        if not len(valid):
            report['stop_reason']='no_valid_frontier';break
        with torch.no_grad():
            q=net(torch.tensor(obs['vector'][None,:],dtype=torch.float32))[0].numpy()
        action=int(valid[np.argmax(q[valid])])
        status('训练后的 D3QN 正在持续探索；不更新权重',
            f'正在执行第 {step+1} 个目标，已运行 {(time.monotonic()-started)/60:.1f} 分钟；上限 20 分钟，无录像。')
        obs,reward,terminated,truncated,info=env.step(action)
        entry=dict(step=step+1,action=action,reward=reward,result=info.get('result'),
            reason=info['reason'],new_free_m2=info.get('new_free_m2'),
            free_area_m2=info.get('ever_free_m2'),duration=info.get('sim_duration'),
            goal=info.get('selected_candidate',{}).get('goal'),valid_transition=info.get('valid_transition'))
        report['steps'].append(entry)
        (a.output/'exploration_progress.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(entry),flush=True)
        failures=0 if info.get('result')=='succeeded' else failures+1
        stagnant=stagnant+1 if info.get('new_free_m2',0)<.05 else 0
        if terminated or truncated:
            report['stop_reason']=info['reason'];break
        if failures>=3:
            report['stop_reason']='three_consecutive_navigation_failures';break
        if stagnant>=10:
            report['stop_reason']='ten_decisions_without_area_gain';break
    else:
        report['stop_reason']='100_decision_safety_limit'
    report['elapsed_seconds']=time.monotonic()-started
    report['completed']=True
except Exception as exc:
    report.update(stop_reason='runtime_error',error=str(exc),traceback=traceback.format_exc())
finally:
    try:
        env.close()
    except Exception as exc:
        report['shutdown_error']=str(exc)
    report['shutdowns']=env.completed_sessions
    report['successes']=sum(s['result']=='succeeded' for s in report['steps'])
    report['non_successes']=len(report['steps'])-report['successes']
    (a.output/'exploration_report.json').write_text(json.dumps(report,indent=2))
    status('本轮探索已结束：'+report.get('stop_reason','unknown'),
        f'执行 {len(report["steps"])} 个目标，到达 {report["successes"]} 次。结果与地图已保存。')
