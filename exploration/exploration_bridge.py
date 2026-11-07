"""Fail closed when ORB changes its active map or global geometry revision."""
import json, time
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from base_navigation import NavigationBridge


class ExplorationBridge(NavigationBridge):
    def __init__(self):
        self.epoch = None
        self.epoch_wall = 0.
        self.invalid_reason = None
        super().__init__()
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, '/exploration/adapter_state', qos)
        self.create_subscription(String, '/slam/map_epoch', self.epoch_cb, qos)
        self.create_timer(.2, self.publish_state)

    def epoch_cb(self, msg):
        value = json.loads(msg.data)
        epoch = (int(value['map_id']), int(value['big_change']))
        self.epoch_wall = time.monotonic()
        if self.epoch is not None and epoch != self.epoch and self.initial_base is not None:
            self.invalidated = True
            self.invalid_reason = 'slam_map_or_global_revision_changed'
            self.odds.fill(0)
            self.known.fill(False)
            self.depths.clear()
            self.raw = Twist()
            self.cmd_pub.publish(Twist())
            if self.active and not self.cancel_pending:
                self.active.cancel_goal_async()
                self.cancel_pending = True
                self.event('map_epoch_cancel_requested')
            self.event('map_epoch_invalidated', previous=list(self.epoch), current=list(epoch))
            self.publish_map()
        self.epoch = epoch
        self.publish_state()

    def healthy(self):
        return (super().healthy() and self.epoch is not None
                and time.monotonic()-self.epoch_wall < .7)

    def publish_state(self):
        self.state_pub.publish(String(data=json.dumps(dict(
            healthy=self.healthy(), invalidated=self.invalidated,
            invalid_reason=self.invalid_reason or ('pose_jump' if self.invalidated else None),
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
