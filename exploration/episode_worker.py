"""Synchronous ROS episode worker behind the Gymnasium file transport.

No Gazebo odometry subscription. The simulator is paused at decision boundaries;
an action indexes the frozen candidate table returned with the last observation.
"""
import hashlib, json, math, os, subprocess, time, traceback
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import ComputePathToPose
from lifecycle_msgs.srv import GetState
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from frontiers import candidates, observation


def atomic(file, value):
    tmp = file.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False))
    tmp.replace(file)


class Worker(Node):
    def __init__(self):
        super().__init__('explorb_episode_worker')
        self.declare_parameter('output_dir', '/results/episode')
        self.out = Path(self.get_parameter('output_dir').value)
        self.protocol = self.out/'protocol'
        self.protocol.mkdir(exist_ok=True)
        self.data = self.base = self.q = self.adapter = None
        self.yaw = 0.
        self.origin = None
        self.res = .05
        self.ever_free = None
        self.events = []
        self.nav_seq = 0
        self.steps = 0
        self.max_steps = int(os.environ.get('EXPLORATION_MAX_STEPS', '4'))
        self.table = []
        self.snapshot = None
        self.paused = False
        self.recent_goals = []
        self.last_command = [0.,0.]
        self.create_subscription(OccupancyGrid, '/map', self.map_cb,
            QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Odometry, '/nav/odometry', self.odom, 20)
        self.create_subscription(String, '/exploration/adapter_state', self.state,
            QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, '/exploration/navigation_status', self.status, 20)
        self.create_subscription(Twist, '/cmd_vel', self.command, 20)
        self.goal_pub = self.create_publisher(PoseStamped, '/exploration/goal', 10)
        self.plan = ActionClient(self, ComputePathToPose, '/compute_path_to_pose')
        self.lifecycle = self.create_client(GetState, '/bt_navigator/get_state')
        self.trace = open(self.out/'transitions.jsonl','w',buffering=1)

    def sim(self):
        return self.get_clock().now().nanoseconds*1e-9

    def map_cb(self, msg):
        if self.adapter and self.adapter['invalidated']:
            return
        self.data = np.array(msg.data,np.int16).reshape(msg.info.height,msg.info.width)
        self.origin = np.array([msg.info.origin.position.x,msg.info.origin.position.y])
        self.res = msg.info.resolution
        if self.ever_free is None:
            self.ever_free = np.zeros_like(self.data,bool)
        self.ever_free |= self.data == 0

    def odom(self, msg):
        p,q = msg.pose.pose.position,msg.pose.pose.orientation
        self.base = np.array([p.x,p.y,p.z])
        self.q = np.array([q.x,q.y,q.z,q.w])
        self.yaw = math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

    def state(self, msg):
        self.adapter = json.loads(msg.data)

    def status(self, msg):
        self.events.append(json.loads(msg.data))

    def command(self, msg):
        self.last_command = [msg.linear.x,msg.angular.z]

    def spin(self, seconds=.05):
        rclpy.spin_once(self,timeout_sec=seconds)

    def wait(self, predicate, timeout, message):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if (self.out/'episode_close.json').exists():
                raise RuntimeError('Episode closed during '+message)
            self.spin()
        raise TimeoutError(message)

    def settle(self, seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            self.spin(.02)

    def pause(self, value):
        result = subprocess.run(['ign','service','-s','/world/aws_small_house/control',
            '--reqtype','ignition.msgs.WorldControl','--reptype','ignition.msgs.Boolean',
            '--timeout','2000','--req','pause: '+str(value).lower()],
            capture_output=True,text=True,timeout=5)
        if result.returncode or 'data: true' not in result.stdout:
            raise RuntimeError('Gazebo pause service failed: '+result.stdout+result.stderr)
        self.paused = value
        self.settle(.25)

    def pose(self, xy, yaw):
        msg = PoseStamped()
        msg.header.frame_id = 'slam_map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x,msg.pose.position.y = map(float,xy)
        msg.pose.orientation.z,msg.pose.orientation.w = math.sin(yaw/2),math.cos(yaw/2)
        return msg

    def plan_valid(self, entry):
        goal = ComputePathToPose.Goal()
        goal.goal = self.pose(entry['goal'],entry['yaw'])
        goal.start = self.pose(self.base[:2],self.yaw)
        goal.use_start = True
        goal.planner_id = 'GridBased'
        future = self.plan.send_goal_async(goal)
        self.wait(future.done,3,'planner acceptance timeout')
        handle = future.result()
        if not handle.accepted:
            return False
        result = handle.get_result_async()
        self.wait(result.done,4,'planner result timeout')
        value = result.result()
        return value.status == 4 and len(value.result.path.poses) > 1

    def build_observation(self):
        self.table, diagnostics = candidates(self.data,self.origin,self.res,self.base,self.yaw)
        self.table = [entry for entry in self.table if self.plan_valid(entry)]
        for entry in self.table:
            entry['visits'] = int(sum(np.linalg.norm(np.array(entry['goal'])-g) < .35 for g in self.recent_goals))
        obs = observation(self.base,self.q,self.yaw,self.table)
        obs = {key:value.tolist() for key,value in obs.items()}
        payload = dict(obs=obs,candidates=self.table,epoch=self.adapter['epoch'],steps=self.steps)
        self.snapshot = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
        return obs, dict(snapshot_id=self.snapshot,candidates=self.table,frontier=diagnostics,
            epoch=self.adapter['epoch'],step=self.steps,sim_time=self.sim(),paused=self.paused,
            ever_free_m2=float(self.ever_free.sum()*self.res*self.res),
            base=self.base.tolist(),valid_transition=True,
            session=self.out.name,observation_source='ORB + RGB-D belief only')

    def initial(self):
        self.wait(lambda:self.data is not None and self.base is not None and self.adapter
                  and self.adapter['healthy'] and self.adapter['map_frames'] >= 8
                  and self.plan.server_is_ready() and self.lifecycle.service_is_ready(),
                  65,'pipeline initialization timeout')
        future = self.lifecycle.call_async(GetState.Request())
        self.wait(future.done,5,'lifecycle query timeout')
        if future.result().current_state.id != 3:
            self.wait(lambda:False,2,'Nav2 not active')
        t = self.sim()
        self.wait(lambda:self.sim()-t >= 1.,5,'initial map settling timeout')
        self.pause(True)
        obs,info = self.build_observation()
        if not any(obs['action_mask']):
            raise RuntimeError('No valid initial frontier approach candidate')
        self.recent_goals.append(self.base[:2].copy())
        atomic(self.protocol/'ready.json',dict(observation=obs,info=info))

    def step(self, request):
        if request['snapshot_id'] != self.snapshot:
            raise RuntimeError('Action refers to a stale candidate table')
        action = request['action']
        self.steps += 1
        if type(action) is not int or not 0 <= action < len(self.table):
            obs = observation(self.base,self.q,self.yaw,self.table)
            obs = {key:value.tolist() for key,value in obs.items()}
            return dict(observation=obs,reward=-.5,terminated=False,
                truncated=self.steps >= self.max_steps,info=dict(snapshot_id=self.snapshot,
                reason='invalid_action',valid_transition=True,step=self.steps,sim_duration=0.,
                candidates=self.table,reward_terms=dict(new_free_m2=0.,time_penalty=0.,failure_penalty=.5)))
        entry = self.table[action].copy()
        before = float(self.ever_free.sum()*self.res*self.res)
        t0 = self.sim()
        self.pause(False)
        warm = self.sim()
        self.wait(lambda:self.adapter['invalidated'] or (
            self.adapter['healthy'] and self.sim()-warm >= .35),8,'sensor refresh timeout')
        result = 'map_invalidated' if self.adapter['invalidated'] else None
        self.nav_seq += 1
        start_events = len(self.events)
        if result is None:
            self.goal_pub.publish(self.pose(entry['goal'],entry['yaw']))
            deadline = time.monotonic()+100
            while time.monotonic() < deadline:
                self.spin()
                if self.adapter['invalidated']:
                    result = 'map_invalidated'
                    break
                for event in self.events[start_events:]:
                    if event['sequence'] != self.nav_seq:
                        continue
                    s = event['status']
                    if s in ('succeeded','failed','canceled') or s.startswith('rejected_'):
                        result = s
                        break
                if result is not None:
                    break
            if result is None:
                # Do not leave a background action running beyond an API timeout.
                raise TimeoutError('Navigation result exceeded wall-time budget')
        if result != 'map_invalidated':
            t = self.sim()
            self.wait(lambda:self.sim()-t >= .7,5,'post-action map settling timeout')
        self.pause(True)
        duration = self.sim()-t0
        sensor_bad = not self.adapter['healthy']
        if self.adapter['invalidated'] or sensor_bad:
            # No old/new-map replay pair is presented as a valid learning sample.
            obs = dict(vector=[0.]*36,action_mask=[0]*10)
            return dict(observation=obs,reward=0.,terminated=False,truncated=True,
                info=dict(reason='slam_map_revision' if self.adapter['invalidated'] else 'tracking_or_sensor_lost',valid_transition=False,
                          epoch=self.adapter['epoch'],snapshot_id=self.snapshot,
                          result=result,step=self.steps))
        gain = max(0.,float(self.ever_free.sum()*self.res*self.res)-before)
        penalty = 0. if result == 'succeeded' else 1.
        reward = gain-.01*duration-penalty
        self.recent_goals.append(np.array(entry['goal']))
        obs,info = self.build_observation()
        no_frontier = not any(obs['action_mask'])
        info.update(result=result,selected_candidate=entry,new_free_m2=gain,sim_duration=duration,
            reason='no_valid_frontier' if no_frontier else ('step_limit' if self.steps >= self.max_steps else result),
            reward_terms=dict(new_free_m2=gain,time_penalty=.01*duration,failure_penalty=penalty))
        self.trace.write(json.dumps(dict(action=action,reward=reward,info=info))+'\n')
        return dict(observation=obs,reward=reward,terminated=no_frontier,
                    truncated=not no_frontier and self.steps >= self.max_steps,info=info)

    def run(self):
        self.initial()
        request_number = 1
        while not (self.out/'episode_close.json').exists():
            self.spin(.03)
            file = self.protocol/f'request_{request_number:04d}.json'
            if not file.exists():
                continue
            request = json.loads(file.read_text())
            if request.get('operation') == 'inspect':
                response = dict(observation={key:value.tolist() for key,value in
                    observation(self.base,self.q,self.yaw,self.table).items()},reward=0.,
                    terminated=False,truncated=False,info=dict(snapshot_id=self.snapshot,
                    sim_time=self.sim(),base=self.base.tolist(),paused=self.paused,
                    step=self.steps,candidates=self.table,valid_transition=True))
            elif request.get('operation') == 'fault_map_revision':
                # Explicit acceptance-only injection; normal step never publishes map metadata.
                epoch = self.adapter['epoch']
                publisher = self.create_publisher(String,'/slam/map_epoch',
                    QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
                self.pause(False)
                self.settle(.5)
                moving = bool(request.get('moving',False))
                start_events = len(self.events)
                if moving:
                    before = self.base.copy()
                    self.nav_seq += 1
                    self.goal_pub.publish(self.pose(self.table[0]['goal'],self.table[0]['yaw']))
                    self.wait(lambda:np.linalg.norm(self.base[:2]-before[:2]) > .06,
                              15,'Fault test did not produce real motion')
                publisher.publish(String(data=json.dumps(dict(map_id=epoch[0],big_change=epoch[1]+1))))
                self.wait(lambda:self.adapter['invalidated'],4,'fault injection was not detected')
                ended = None
                if moving:
                    self.wait(lambda:any(e['sequence']==self.nav_seq and e['status'] in ('canceled','failed')
                        for e in self.events[start_events:]),5,'Map revision did not end active navigation')
                    ended = next(e['status'] for e in self.events[start_events:]
                        if e['sequence']==self.nav_seq and e['status'] in ('canceled','failed'))
                self.settle(.3)
                self.pause(True)
                response = dict(observation=dict(vector=[0.]*36,action_mask=[0]*10),reward=0.,
                    terminated=False,truncated=True,info=dict(reason='slam_map_revision',
                    valid_transition=False,fault_injected=True,snapshot_id=self.snapshot,epoch=self.adapter['epoch']))
                response['info'].update(moving=moving,navigation_result=ended,
                    known_cells=self.adapter['known_cells'],command=self.last_command)
                self.destroy_publisher(publisher)
            else:
                response = self.step(request)
            atomic(self.protocol/f'response_{request_number:04d}.json',response)
            request_number += 1


if __name__ == '__main__':
    rclpy.init()
    node = Worker()
    try:
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        atomic(node.out/'fatal.json',dict(error=str(exc),traceback=traceback.format_exc()))
        while rclpy.ok():
            node.spin(.1)
    finally:
        node.trace.close()
        node.destroy_node()
        rclpy.try_shutdown()
