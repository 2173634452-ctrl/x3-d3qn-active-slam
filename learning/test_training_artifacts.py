"""Check durable diagnostics and distinct checkpoints on a tiny synthetic replay."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import torch

with tempfile.TemporaryDirectory() as folder:
    root=Path(folder)
    dataset=root/'data.jsonl'
    row=dict(state=[0.]*36,action_mask=[1]*10,action=0,reward=1.,next_state=[.1]*36,
        next_action_mask=[1]*10,terminated=False,truncated=True,duration=10.)
    dataset.write_text((json.dumps(row)+'\n')*8)
    out=root/'train'
    subprocess.run([sys.executable,str(Path(__file__).with_name('train_offline.py')),
        '--dataset',str(dataset),'--output',str(out),'--updates','8','--checkpoint-every','4'],
        check=True,stdout=subprocess.DEVNULL)
    metrics=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    assert len(metrics)==8
    for key in ['loss','td_abs_mean','q_mean','q_abs_max','target_mean','value_grad_norm','advantage_grad_norm']:
        assert all(math.isfinite(m[key]) for m in metrics),key
    assert len(list((out/'checkpoints').glob('*.pt')))==3
    initial=torch.load(out/'checkpoints/update_00000000.pt',weights_only=True)
    final=torch.load(out/'checkpoints/update_00000008.pt',weights_only=True)
    assert any(not torch.equal(initial['policy'][key],final['policy'][key]) for key in final['policy'])
    assert final['updates']==8
    print(json.dumps(dict(passed=True,synthetic_replay=True,updates=8,
        checks=['durable per-update diagnostics','initial and periodic checkpoints','parameters changed'])))
