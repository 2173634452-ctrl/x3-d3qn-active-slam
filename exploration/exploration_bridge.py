"""Rebuild the RGB-D grid when the active ORB map is globally corrected."""
from collections import deque
import json
import time
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from base_navigation import NavigationBridge, rotation, stamp


class ExplorationBridge(NavigationBridge):
    def __init__(self):
        self.epoch = None
        self.epoch_wall = 0.
        self.invalid_reason = None
        self.anchors = {}
        self.anchor_epoch = None
        self.rebuild_pending = False
        self.rebuilt_epoch = None
        self.cancel_on_accept = False
        self.scans = deque(maxlen=3000)
        super().__init__()
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, '/exploration/adapter_state', qos)
        self.create_subscription(String, '/slam/map_epoch', self.epoch_cb, qos)
        self.create_subscription(Path, '/slam/map_anchors', self.anchors_cb, qos)
        self.create_timer(.2, self.publish_state)

    def epoch_cb(self, msg):
        value = json.loads(msg.data)
        epoch = (int(value['map_id']), int(value['big_change']))
        self.epoch_wall = time.monotonic()
        if self.epoch is not None and epoch != self.epoch and self.initial_base is not None:
            self.invalidated = True
            self.invalid_reason = 'slam_map_or_global_revision_changed'
            self.rebuild_pending = epoch[0] == self.epoch[0] and epoch[1] > self.epoch[1]
            if epoch[0] != self.epoch[0]:
                self.scans.clear()
            self.odds.fill(0)
            self.known.fill(False)
            self.depths.clear()
            self.poses.clear()
            self.pose = None
            self.raw = Twist()
            self.cmd_pub.publish(Twist())
            self.cancel_on_accept = self.pending
            if self.active and not self.cancel_pending:
                self.active.cancel_goal_async()
                self.cancel_pending = True
                self.event('map_epoch_cancel_requested')
            self.event('map_epoch_invalidated', previous=list(self.epoch), current=list(epoch))
            self.publish_map()
        self.epoch = epoch
        self.publish_state()

    def accepted(self, future):
        super().accepted(future)
        if self.cancel_on_accept and self.active:
            self.active.cancel_goal_async()
            self.cancel_pending = True
            self.event('map_epoch_cancel_requested')
        self.cancel_on_accept = False

    def attach_anchors(self, scan):
        scan_time, _, camera, rot, references = scan
        if references or not self.anchors:
            return
        nearest = sorted(self.anchors.items(), key=lambda item: abs(item[1][0]-scan_time))[:3]
        for key, (_, position, orientation) in nearest:
            references.append((key, orientation.T @ (camera-position), orientation.T @ rot))

    def remember_scan(self, scan_time, points, camera, rot):
        scan = (scan_time, points.astype(np.float32), camera, rot, [])
        if self.anchor_epoch == self.epoch:
            self.attach_anchors(scan)
        self.scans.append(scan)

    def anchors_cb(self, msg):
        epoch = tuple(map(int, msg.header.frame_id.split(':')))
        if epoch != self.epoch:
            return
        self.anchor_epoch = epoch
        self.anchors = {
            pose.header.frame_id: (
                stamp(pose),
                np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]),
                rotation(pose.pose.orientation),
            ) for pose in msg.poses
        }
        if self.rebuild_pending:
            self.rebuild()
        elif not self.invalidated:
            for scan in reversed(self.scans):
                if scan[-1]:
                    break
                self.attach_anchors(scan)

    def rebuild(self):
        start = time.monotonic()
        replayed = 0
        for _, points, _, _, references in self.scans:
            for key, offset, local_rotation in references:
                if key not in self.anchors:
                    continue
                _, position, orientation = self.anchors[key]
                self.integrate_scan(points, position + orientation @ offset, orientation @ local_rotation)
                replayed += 1
                break
        if not replayed:
            self.rebuild_pending = False
            self.event('map_epoch_rebuild_unavailable', epoch=list(self.epoch))
            return
        self.rebuild_pending = False
        self.rebuilt_epoch = self.epoch
        self.invalidated = False
        self.invalid_reason = None
        # Require a new live depth update before issuing motion.
        self.poses.clear()
        self.depths.clear()
        self.pose = None
        self.depth_wall = 0.
        self.event('map_epoch_rebuilt', epoch=list(self.epoch), replayed_scans=replayed,
            retained_scans=len(self.scans), seconds=time.monotonic()-start)
        self.publish_state()
        self.publish_map()

    def healthy(self):
        return (super().healthy() and self.epoch is not None
                and time.monotonic()-self.epoch_wall < .7)

    def publish_state(self):
        self.state_pub.publish(String(data=json.dumps(dict(
            healthy=self.healthy(), invalidated=self.invalidated,
            invalid_reason=self.invalid_reason or ('pose_jump' if self.invalidated else None),
            rebuild_pending=self.rebuild_pending,
            rebuilt_epoch=list(self.rebuilt_epoch) if self.rebuilt_epoch is not None else None,
            navigation_idle=self.active is None and not self.pending,
            epoch=list(self.epoch) if self.epoch is not None else None,
            map_frames=self.map_frames, sequence=self.seq, sim_time=self.now_s(),
            known_cells=int(self.known.sum()),tracking_state=self.state,
            pose_age_wall_s=time.monotonic()-self.pose_wall,
            depth_age_wall_s=time.monotonic()-self.depth_wall,
            base=self.base.tolist() if self.base is not None else None))))


if __name__ == '__main__':
    rclpy.init()
    node = ExplorationBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
        node.log.close()
        node.commands.close()
        node.destroy_node()
        rclpy.try_shutdown()
