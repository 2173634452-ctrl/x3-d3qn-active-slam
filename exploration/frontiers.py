"""Belief-only frontier approach candidates with a fixed, padded action table."""
from collections import deque
import math
import cv2
import numpy as np


def candidates(data, origin, res, base, yaw, k=10, lookahead=.85):
    free, unknown = data == 0, data < 0
    radius = int(math.ceil(.19/res))
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk = (xx*xx+yy*yy <= radius*radius).astype(np.uint8)
    occupied = cv2.dilate((data == 100).astype(np.uint8), disk) > 0
    traversable = free & ~occupied
    safe_goal = cv2.erode(free.astype(np.uint8), disk, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    edge = free & (cv2.dilate(unknown.astype(np.uint8), np.ones((3,3),np.uint8)) > 0)
    h, w = data.shape
    start = np.floor((np.asarray(base[:2])-origin)/res).astype(int)
    sx, sy = start
    if not (0 <= sx < w and 0 <= sy < h) or not traversable[sy, sx]:
        return [], dict(frontier_cells=int(edge.sum()), reachable_cells=0)
    # Four-neighbor paths avoid diagonal corner cutting. Unknown centers and the
    # occupied footprint inflation are excluded, as in the navigation contract.
    dist = np.full((h,w), -1, np.int32)
    parent = np.full((h,w,2), -1, np.int16)
    dist[sy,sx] = 0
    queue = deque([(sy,sx)])
    while queue:
        y,x = queue.popleft()
        for dy,dx in ((0,1),(1,0),(0,-1),(-1,0)):
            ny,nx = y+dy,x+dx
            if 0 <= ny < h and 0 <= nx < w and traversable[ny,nx] and dist[ny,nx] < 0:
                dist[ny,nx] = dist[y,x]+1
                parent[ny,nx] = y,x
                queue.append((ny,nx))
    safe = np.argwhere(safe_goal & (dist > 0))
    edges = np.argwhere(edge & (dist >= 0))
    if len(safe) == 0 or len(edges) == 0:
        return [], dict(frontier_cells=int(edge.sum()), reachable_cells=int((dist>=0).sum()))
    bucket_size = max(1,int(.60/res))
    groups = {}
    for y,x in edges:
        groups.setdefault((y//bucket_size,x//bucket_size), []).append((y,x))
    proposals = []
    for group in groups.values():
        if len(group) < 3:
            continue
        # A mean can land between two different frontier boundaries. Select an
        # actual frontier cell (medoid proxy), so the heading points at an edge.
        group = np.asarray(group)
        centroid = group[np.argmin(np.square(group-np.mean(group,axis=0)).sum(axis=1))]
        d2 = np.square(safe-centroid).sum(axis=1)
        i = int(np.argmin(d2))
        if d2[i]*res*res > .8**2:
            continue
        y,x = map(int,safe[i])
        path = [(y,x)]
        while dist[y,x] > 0:
            y,x = map(int,parent[y,x])
            path.append((y,x))
        path.reverse()
        usable = [(j,(y,x)) for j,(y,x) in enumerate(path)
                  if .30 <= j*res <= lookahead and safe_goal[y,x]]
        if not usable:
            continue
        j,(gy,gx) = usable[-1]
        xy = origin+(np.array([gx,gy])+.5)*res
        bearing = math.atan2(xy[1]-base[1],xy[0]-base[0])
        relative_bearing = math.atan2(math.sin(bearing-yaw),math.cos(bearing-yaw))
        # The current low-level interface forbids reverse motion and has only a
        # forward depth camera. Keep each short action in a forward 138° sector;
        # gradual end-heading changes reveal new directions on later decisions.
        if abs(relative_bearing) > 1.20:
            continue
        frontier_xy = origin+(centroid[::-1]+.5)*res
        heading = math.atan2(frontier_xy[1]-xy[1], frontier_xy[0]-xy[0])
        turn = math.atan2(math.sin(heading-yaw),math.cos(heading-yaw))
        heading = yaw+float(np.clip(turn,-.65,.65))
        rg = int(.7/res)
        y0,y1 = max(0,gy-rg),min(h,gy+rg+1)
        x0,x1 = max(0,gx-rg),min(w,gx+rg+1)
        vv,uu = np.mgrid[y0:y1,x0:x1]
        around = (vv-gy)**2+(uu-gx)**2 <= rg*rg
        # This is local unknown area potential, not a true sensor entropy/D-opt score.
        gain = float(np.count_nonzero(unknown[y0:y1,x0:x1] & around)*res*res)
        proposals.append(dict(goal=xy.tolist(), yaw=heading, gain_m2=gain,
                              path_m=float(j*res), frontier=frontier_xy.tolist(),
                              score=gain/(.3+j*res)))
    proposals.sort(key=lambda c:(-c['score'],c['goal'][0],c['goal'][1]))
    chosen = []
    for proposal in proposals:
        if all(np.linalg.norm(np.array(proposal['goal'])-c['goal']) > .22 for c in chosen):
            chosen.append(proposal)
        if len(chosen) == k:
            break
    return chosen, dict(frontier_cells=int(edge.sum()), reachable_cells=int((dist>=0).sum()),
                        safe_goal_cells=int(len(safe)), raw_proposals=len(proposals))


def observation(base, quaternion, yaw, table, k=10):
    vector = np.zeros(6+3*k, np.float32)
    vector[:2] = np.clip(np.asarray(base[:2])/8.,-1,1)
    vector[2:6] = quaternion
    c,s = math.cos(yaw),math.sin(yaw)
    rot = np.array([[c,s],[-s,c]])
    mask = np.zeros(k, np.int8)
    for i,entry in enumerate(table):
        vector[6+2*i:8+2*i] = np.clip(rot @ (np.array(entry['goal'])-base[:2])/4.,-1,1)
        vector[6+2*k+i] = min(1., entry['gain_m2']/(math.pi*.7**2))
        mask[i] = 1
    return dict(vector=vector, action_mask=mask)
