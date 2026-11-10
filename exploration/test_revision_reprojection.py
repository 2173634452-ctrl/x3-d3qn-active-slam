"""Exercise scan replay with a moved and rotated keyframe, without Gazebo."""
from collections import deque
from types import SimpleNamespace

import numpy as np

from exploration_bridge import ExplorationBridge


def test_scan_replayed_in_corrected_frame():
    points = np.array([[1., 0., 0.]])
    old_position = np.array([2., 0., 0.])
    camera = np.array([3., 0., 0.])
    identity = np.eye(3)
    scan = (1., points, camera, identity, [])
    poses = []
    events = []
    bridge = SimpleNamespace(
        scans=deque([scan]), anchors={'3': (1., old_position, identity)},
        epoch=(0, 1), rebuild_pending=True, invalidated=True,
        poses=deque(), depths=deque(), pose=None,
        integrate_scan=lambda pts, cam, rot: poses.append(pts @ rot.T + cam),
        event=lambda name, **details: events.append((name, details)),
        publish_state=lambda: None, publish_map=lambda: None,
    )
    ExplorationBridge.attach_anchors(bridge, scan)
    rotated = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    bridge.anchors = {'3': (1., np.array([4., 2., 0.]), rotated)}
    ExplorationBridge.rebuild(bridge)
    np.testing.assert_allclose(poses[0], [[4., 4., 0.]])
    assert bridge.rebuilt_epoch == (0, 1) and not bridge.invalidated
    assert events[-1][1]['replayed_scans'] == 1


if __name__ == '__main__':
    test_scan_replayed_in_corrected_frame()
    print('Corrected scan translation and rotation passed')
