"""Geometry/observation contract checks, including disconnected and thin free space."""
import json
import cv2
import numpy as np
from frontiers import candidates, observation

origin = np.array([-4.,-4.])
res = .05
data = np.full((160,160),-1,np.int16)
data[40:120,40:120] = 0
data[40:120,100] = 100
table,info = candidates(data,origin,res,np.array([0.,0.,0.]),0.)
assert table, info
assert len(table) <= 10
for entry in table:
    x,y = np.floor((np.array(entry['goal'])-origin)/res).astype(int)
    assert x < 100, 'A candidate crossed a separating wall'
    assert np.all(data[y-4:y+5,x-4:x+5] != 100), 'Goal overlaps inflated obstacle'
    fx,fy = np.floor((np.array(entry['frontier'])-origin)/res).astype(int)
    assert data[fy,fx] == 0 and np.any(data[fy-1:fy+2,fx-1:fx+2] < 0), 'Frontier proxy is not on a frontier'
obs = observation(np.zeros(3),np.array([0.,0.,0.,1.]),0.,table)
assert obs['vector'].shape == (36,) and obs['vector'].dtype == np.float32
assert obs['action_mask'].shape == (10,) and obs['action_mask'].sum() == len(table)
assert np.all(np.isfinite(obs['vector'])) and np.max(np.abs(obs['vector'])) <= 1.
assert np.all(obs['vector'][6+2*len(table):26] == 0)
assert np.all(obs['vector'][26+len(table):] == 0)
unknown = np.full_like(data,-1)
assert not candidates(unknown,origin,res,np.zeros(3),0.)[0]
thin = unknown.copy()
thin[79:82,40:120] = 0
assert not candidates(thin,origin,res,np.zeros(3),0.)[0], 'Narrow corridor cannot fit the robot'
print(json.dumps(dict(passed=True,checks=['disconnected free regions','actual frontier cell',
    'obstacle clearance','fixed observation and padding','unknown map','thin corridor'],
    candidate_count=len(table))))
