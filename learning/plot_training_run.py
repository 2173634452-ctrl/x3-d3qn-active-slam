"""Show update diagnostics and independently recorded loaded-policy motion."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
a=p.parse_args()
r=a.run
report=json.loads((r/'training_acceptance.json').read_text())
updates=report['updates']
episode=next(x for x in (r/'evaluation/simulation').glob('episode_*') if x.is_dir())
rows=list(csv.DictReader((episode/'truth_odometry.csv').open()))
xy=np.array([[float(e['x']),float(e['y'])] for e in rows])
xy-=xy[0]
ready=json.loads((episode/'protocol/ready.json').read_text())
base=np.array(ready['info']['base'][:2])
goals=[]
for file in sorted((episode/'protocol').glob('response_*.json')):
    response=json.loads(file.read_text())
    goals.append(np.array(response['info']['selected_candidate']['goal'])-base)
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
axes[0].plot([m['update'] for m in updates],[m['loss'] for m in updates],marker='o')
axes[0].set(xlabel='Optimization update',ylabel='Smooth L1 TD loss',
    title='12 updates: pipeline diagnostics, not convergence')
axes[1].plot(xy[:,0],xy[:,1],linewidth=2,label='Gazebo truth (post-run only)')
axes[1].scatter(0,0,color='green',label='Start')
for i,goal in enumerate(goals):
    axes[1].scatter(*goal,marker='*',s=100,color='orange')
    axes[1].annotate(f'Goal {i+1}',goal,xytext=(6,6),textcoords='offset points')
axes[1].set(xlabel='Forward from initial pose / m',ylabel='Left from initial pose / m',
    title='Loaded checkpoint: two greedy navigation goals',aspect='equal')
axes[1].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(r/'training_and_inference.png',dpi=160)
