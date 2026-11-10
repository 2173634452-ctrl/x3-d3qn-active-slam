#!/usr/bin/env python3
"""RGB-D / ORB-only navigation adapter. No simulator truth input.

Short-run integration baseline: costmaps use slam_map directly. It is deliberately
not a production map->odom fusion solution. A pose discontinuity invalidates the
accumulated depth map and stops navigation instead of silently reusing stale cells.
"""
import json, math, time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32, String
from cv_bridge import CvBridge


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class NavigationBridge(Node):
    def __init__(self):
        super().__init__('explorb_navigation_bridge')
        self.declare_parameter('output_dir', '/results/nav')
        self.out = Path(self.get_parameter('output_dir').value)
        self.out.mkdir(exist_ok=True, parents=True)
        self.log = open(self.out/'navigation_events.jsonl', 'w', buffering=1)
        self.commands = open(self.out/'commands.jsonl', 'w', buffering=1)
        self.res, self.n = .05, 320
        self.origin = -self.res*self.n/2
        self.odds = np.zeros((self.n, self.n), np.float32)
        self.known = np.zeros((self.n, self.n), bool)
        self.poses = deque(maxlen=80)
        self.depths = deque(maxlen=10)
        self.k = None
        self.state, self.state_wall = 0, 0
        self.pose_wall = self.depth_wall = 0
        self.pose = self.base = self.initial_base = None
        self.last_map_stamp = -1
        self.last_depth_stamp = -1
        self.map_frames = 0
        self.invalidated = False
        self.raw, self.raw_wall = Twist(), 0
        self.front_clearance = float('inf')
        self.active = None
        self.pending = False
        self.active_start = 0
        self.seq = 0
        self.request_counter = 0
        self.cancel_pending = False
        self.unhealthy_since = None
        self.nonzero = self.gated = 0
        self.bridge = CvBridge()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', latched)
        self.odom_pub = self.create_publisher(Odometry, '/nav/odometry', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/exploration/navigation_status', 20)
        self.action = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.create_subscription(CameraInfo, '/camera/info', self.info, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth', self.depth, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/slam/odometry', self.odom, 20)
        self.create_subscription(Int32, '/slam/tracking_state', self.tracking, 20)
        self.create_subscription(Twist, '/nav/raw_cmd_vel', self.raw_command, 10)
        self.create_subscription(PoseStamped, '/exploration/goal', self.goal, 10)
        self.create_timer(.1, self.tick)
        self.create_timer(.5, self.publish_map)
        self.event('adapter_started', localization='/slam/odometry', obstacles='/camera/depth',
                   frame='slam_map', resolution=self.res, ground_truth_used=False)

    def now_s(self):
        return self.get_clock().now().nanoseconds*1e-9

    def event(self, status, **extra):
        entry = dict(status=status, sim_time=self.now_s(), sequence=extra.pop('sequence', self.seq), **extra)
        self.log.write(json.dumps(entry, allow_nan=False)+'\n')
        self.status_pub.publish(String(data=json.dumps(entry)))

    def info(self, m):
        self.k = np.array(m.k).reshape(3, 3)

    def tracking(self, m):
        self.state, self.state_wall = m.data, time.monotonic()

    def odom(self, m):
        if m.header.frame_id != 'slam_map' or m.child_frame_id != 'slam_camera':
            return
        rot = rotation(m.pose.pose.orientation)
        cam = np.array([m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z])
        base = cam - rot @ np.array([.057105, .000017948, .11905])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        t = stamp(m)
        out = Odometry()
        out.header = m.header
        out.child_frame_id = 'slam_base_footprint'
        out.pose.pose = m.pose.pose
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = map(float, base)
        if self.pose is not None:
            dt = t-self.pose[0]
            delta = base-self.base
            dyaw = math.atan2(math.sin(yaw-self.pose[3]), math.cos(yaw-self.pose[3]))
            if 0 < dt < .5:
                if np.linalg.norm(delta[:2]) > .25 or abs(dyaw) > .35:
                    self.invalidated = True
                    self.odds.fill(0)
                    self.known.fill(False)
                    self.event('map_invalidated_pose_jump')
                else:
                    body = rot.T @ (delta/dt)
                    out.twist.twist.linear.x = float(body[0])
                    out.twist.twist.linear.y = float(body[1])
                    out.twist.twist.angular.z = dyaw/dt
        self.base = base
        self.pose = (t, cam, rot, yaw)
        self.poses.append(self.pose)
        self.pose_wall = time.monotonic()
        if self.initial_base is None:
            self.initial_base = base.copy()
            self.event('first_base_pose', position=base.tolist())
            # The currently occupied footprint is the only seeded free-space prior.
            mask = np.zeros_like(self.odds, np.uint8)
            cv2.circle(mask, self.cell(base[:2]), 4, 1, -1)
            self.known[mask > 0] = True
            self.odds[mask > 0] = -3
        self.odom_pub.publish(out)

    def cell(self, xy):
        return (int(math.floor((float(xy[0])-self.origin)/self.res)),
                int(math.floor((float(xy[1])-self.origin)/self.res)))

    def depth(self, m):
        self.depths.append(m)

    def integrate(self):
        if self.k is None or not self.poses or not self.depths or self.state != 2 or self.invalidated:
            return
        # ORB runs behind the camera. Pair a buffered depth with an estimated pose
        # at the SAME image timestamp; never transform a fresh depth using an old pose.
        best = None
        for m in reversed(self.depths):
            t = stamp(m)
            if t <= self.last_depth_stamp+.199:
                continue
            pose = min(self.poses, key=lambda p: abs(p[0]-t))
            if abs(pose[0]-t) < .012:
                best = (m, pose)
                break
        if best is None:
            return
        m, (t, cam, rot, yaw) = best
        depth = self.bridge.imgmsg_to_cv2(m, 'passthrough')
        if m.encoding == '16UC1':
            depth = depth.astype(np.float32)*.001
        elif m.encoding != '32FC1':
            return
        v, u = np.mgrid[0:m.height:8, 0:m.width:8]
        d = depth[v, u]
        valid = np.isfinite(d) & (d > .07) & (d < 5.)
        d, u, v = d[valid], u[valid], v[valid]
        # Optical right/down/forward -> camera forward/left/up, same S as ORB wrapper.
        local = np.column_stack((d, -(u-self.k[0, 2])*d/self.k[0, 0],
                                 -(v-self.k[1, 2])*d/self.k[1, 1]))
        self.integrate_scan(local, cam, rot)
        self.remember_scan(t, local, cam, rot)
        self.last_depth_stamp = t
        self.depth_wall = time.monotonic()
        self.map_frames += 1

    def remember_scan(self, stamp, points, camera, rotation):
        """Hook for a mapper that retains scans for pose correction."""

    def integrate_scan(self, local, cam, rot):
        pts = local @ rot.T + cam
        base = cam - rot @ np.array([.057105, .000017948, .11905])
        # A low camera cannot detect every overhang. This slab is for X3's current
        # body/scene only. Keep floor returns free, and obstacle endpoints occupied.
        relz = pts[:, 2]-base[2]
        use = (relz < .45) & (relz > -.12)
        pts, relz = pts[use], relz[use]
        hit = relz > .035
        body = (pts-base) @ rot
        corridor = (body[:, 0] > 0) & (np.abs(body[:, 1]) < .18) & hit
        self.front_clearance = float(body[corridor, 0].min()) if np.any(corridor) else float('inf')
        free, occupied = np.zeros_like(self.odds, np.uint8), np.zeros_like(self.odds, np.uint8)
        start = self.cell(cam[:2])
        for xy, obstacle in zip(pts[:, :2], hit):
            end = self.cell(xy)
            if not (0 <= end[0] < self.n and 0 <= end[1] < self.n):
                continue
            cv2.line(free, start, end, 1, 1)
            if obstacle:
                occupied[end[1], end[0]] = 1
        self.known[(free > 0) | (occupied > 0)] = True
        self.odds[(free > 0) & (occupied == 0)] -= .35
        self.odds[occupied > 0] += 1.1
        np.clip(self.odds, -3, 3, out=self.odds)

    def publish_map(self):
        if self.initial_base is None:
            return
        data = np.full((self.n, self.n), -1, np.int8)
        data[self.known & (self.odds < -.5)] = 0
        data[self.known & (self.odds > .5)] = 100
        m = OccupancyGrid()
        m.header.frame_id = 'slam_map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution, m.info.width, m.info.height = self.res, self.n, self.n
        m.info.origin.position.x = m.info.origin.position.y = self.origin
        m.info.origin.orientation.w = 1.
        m.data = data.ravel().tolist()
        self.map_pub.publish(m)
        np.save(self.out/'occupancy.npy', data)
        cv2.imwrite(str(self.out/'map.png'), np.flipud(np.where(data < 0, 128, 255-data*2.55).astype(np.uint8)))
        (self.out/'adapter_summary.json').write_text(json.dumps(dict(
            map_frames=self.map_frames, known_free_cells=int(np.count_nonzero(data == 0)),
            occupied_cells=int(np.count_nonzero(data == 100)), nonzero_commands=self.nonzero,
            gated_commands=self.gated, map_invalidated=self.invalidated,
            base_position=self.base.tolist(), resolution=self.res, origin=self.origin,
            truth_subscriptions=[]), indent=2))

    def raw_command(self, m):
        self.raw, self.raw_wall = m, time.monotonic()

    def healthy(self):
        now = time.monotonic()
        return (self.state == 2 and now-self.state_wall < .6 and now-self.pose_wall < .6
                and now-self.depth_wall < .7 and not self.invalidated)

    def goal(self, m):
        self.request_counter += 1
        if self.active is not None or self.pending:
            self.event('rejected_busy', sequence=self.request_counter)
            return
        self.seq = self.request_counter
        if not self.healthy():
            self.event('rejected_tracking_or_sensor_unavailable')
            return
        if m.header.frame_id != 'slam_map':
            self.event('rejected_frame', expected='slam_map')
            return
        x, y = self.cell([m.pose.position.x, m.pose.position.y])
        self.event('goal_received', goal=[m.pose.position.x, m.pose.position.y])
        radius = int(math.ceil(.19/self.res))
        if not (radius <= x < self.n-radius and radius <= y < self.n-radius):
            self.event('rejected_outside_map')
            return
        # Unknown cells are never accepted as valid target positions.
        area = self.odds[y-radius:y+radius+1, x-radius:x+radius+1]
        known = self.known[y-radius:y+radius+1, x-radius:x+radius+1]
        yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        footprint = xx*xx+yy*yy <= radius*radius
        if np.any((area > .5) & known & footprint):
            self.event('rejected_obstacle')
            return
        if np.any((~known) & footprint) or np.any((area >= -.5) & footprint):
            self.event('rejected_unknown')
            return
        if not self.action.server_is_ready():
            self.event('rejected_nav2_unavailable')
            return
        self.pending = True
        req = NavigateToPose.Goal()
        req.pose = m
        req.pose.header.stamp = self.get_clock().now().to_msg()
        self.action.send_goal_async(req).add_done_callback(self.accepted)

    def accepted(self, future):
        self.pending = False
        self.active = future.result()
        if not self.active.accepted:
            self.active = None
            self.event('rejected_by_nav2')
            return
        self.active_start = self.now_s()
        self.cancel_pending = False
        self.event('executing')
        self.active.get_result_async().add_done_callback(self.finished)

    def finished(self, future):
        status = future.result().status
        self.active = None
        self.cancel_pending = False
        self.raw = Twist()
        self.cmd_pub.publish(Twist())
        self.event({4:'succeeded', 5:'canceled', 6:'failed'}.get(status, 'unknown_result'), action_status=status)

    def tick(self):
        self.integrate()
        if self.healthy():
            self.unhealthy_since = None
        elif self.unhealthy_since is None:
            self.unhealthy_since = self.now_s()
        lost_too_long = self.unhealthy_since is not None and self.now_s()-self.unhealthy_since > 2.
        if self.active and not self.cancel_pending and (self.invalidated or lost_too_long or self.now_s()-self.active_start > 50):
            self.event('cancel_requested', reason='tracking_or_sensor_lost' if not self.healthy() else 'timeout')
            self.active.cancel_goal_async()
            # Do not repeat cancellation requests every control tick.
            self.cancel_pending = True
        cmd = Twist()
        valid = self.healthy() and not self.cancel_pending and time.monotonic()-self.raw_wall < .4 and self.active is not None
        if valid:
            cmd.linear.x = max(0., min(.12, self.raw.linear.x))
            cmd.angular.z = max(-.25, min(.25, self.raw.angular.z))
            if self.front_clearance < .28:
                cmd.linear.x = 0.
        if abs(cmd.linear.x)+abs(cmd.angular.z) > 1e-5:
            self.nonzero += 1
        elif abs(self.raw.linear.x)+abs(self.raw.angular.z) > 1e-5:
            self.gated += 1
        self.cmd_pub.publish(cmd)
        self.commands.write(json.dumps(dict(sim_time=self.now_s(), vx=cmd.linear.x, wz=cmd.angular.z,
                                           healthy=self.healthy(), state=self.state))+'\n')


def main():
    rclpy.init()
    node = NavigationBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
            node.publish_map()
        node.log.close()
        node.commands.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
