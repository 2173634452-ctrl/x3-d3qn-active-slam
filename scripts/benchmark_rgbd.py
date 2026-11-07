#!/usr/bin/env python3
"""Bounded, isolated live RGB-D benchmark. Stops only its own processes."""
from pathlib import Path
import argparse,csv,json,os,signal,subprocess,time,shutil,math
from collections import Counter
import cv2,numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,CameraInfo
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path as RosPath
from tf2_msgs.msg import TFMessage
from rclpy.qos import QoSProfile,DurabilityPolicy
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from cv_bridge import CvBridge

p=argparse.ArgumentParser();p.add_argument('--localization-map-id',type=int,default=-1);p.add_argument('--diagnostics',action='store_true');p.add_argument('--record-bag',action='store_true');p.add_argument('--wall-limit',type=float,default=100);p.add_argument('--sim-duration',type=float,default=25);p.add_argument('--no-slam',action='store_true');p.add_argument('--save-atlas',action='store_true');p.add_argument('--load-atlas',type=Path);p.add_argument('--recovery',action='store_true');p.add_argument('--look-room',action='store_true');p.add_argument('--waypoints',type=Path);p.add_argument('--speed',type=float,default=.20);p.add_argument('--camera-height',type=float,default=.11905);p.add_argument('--camera-fps',type=float,default=5);p.add_argument('--viewer',action='store_true');p.add_argument('--route',choices=['forward','rectangle','spin','stationary','demo','tour'],default='forward');p.add_argument('--world',type=Path);p.add_argument('--session-dir',type=Path);a=p.parse_args()
if a.recovery and a.route!='forward':p.error('--recovery requires --route forward')
r=Path(__file__).resolve().parents[1];out=a.session_dir.resolve() if a.session_dir else r/'reports'/time.strftime('rgbd_%Y%m%d_%H%M%S');out.mkdir(parents=True)
for src in ['worlds/x3_house_rgbd.sdf','worlds/x3_rgbd.urdf','worlds/x3_rgbd.sdf','config/x3_rgbd.yaml','runtime/rgbd_node.cpp','scripts/benchmark_rgbd.py']:
 shutil.copy2(r/src,out/Path(src).name)
world=a.world.resolve() if a.world else r/'worlds/x3_house_rgbd.sdf'
if a.world:shutil.copy2(world,out/'presentation_world.sdf')
os.environ.setdefault('ROS_DOMAIN_ID','84');env=dict(os.environ,PYTHONNOUSERSITE='1');env.setdefault('IGN_PARTITION','x3_rgbd_benchmark')
tour=json.loads(a.waypoints.read_text())['waypoints'] if a.waypoints else [];goal_index=1;truth_pose=None;lost_since=None;current_state=0;look_yaw=0.
if a.route=='tour' and not tour:raise RuntimeError('--tour requires --waypoints')
processes=[];logs=[];streams={};clock=[];camera_info={};first_images={};states=Counter();clock_now=0.;last_command=-1.;slam_started=None;tracking_first=None;route_start=None;events=[];localization_on=False;black_on=False;yaw_previous=None;yaw_total=0.;yaw_start=0.
rclpy.init();node=rclpy.create_node('rgbd_benchmark');bridge=CvBridge();commands=node.create_publisher(Twist,'/cmd_vel',10)
blackpub=node.create_publisher(Bool,'/slam/blackout',10);modepub=node.create_publisher(Bool,'/slam/localization_only',10)
published_outputs={};latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
def map_cb(msg):published_outputs['map_points']={'count':msg.width*msg.height,'frame':msg.header.frame_id}
def path_cb(msg,name):published_outputs[name]={'poses':len(msg.poses),'frame':msg.header.frame_id}
def tf_cb(msg):
 for t in msg.transforms:published_outputs.setdefault('tf',{})[t.child_frame_id]={'parent':t.header.frame_id,'qnorm':sum(getattr(t.transform.rotation,k)**2 for k in ['x','y','z','w'])**.5}
stream_file=open(out/'streams.csv','w');writer=csv.writer(stream_file);writer.writerow(['stream','wall_monotonic','stamp'])
odom_file=open(out/'truth_odometry.csv','w');ow=csv.writer(odom_file);ow.writerow(['stamp','x','y','z','qx','qy','qz','qw'])
started=time.monotonic()
def launch(name,cmd):
 log=open(out/(name+'.log'),'w');logs.append(log);proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=out);processes.append((name,proc));return proc
def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
def record(kind,t):
 now=time.monotonic();streams.setdefault(kind,[]).append([now,t]);writer.writerow([kind,now,t])
def image(msg,kind):
 record(kind,stamp(msg))
 if kind not in first_images:
  first_images[kind]={'encoding':msg.encoding,'width':msg.width,'height':msg.height,'stamp':stamp(msg),'frame_id':msg.header.frame_id}
  mat=bridge.imgmsg_to_cv2(msg,'passthrough' if kind=='depth' else 'bgr8')
  if kind=='rgb':cv2.imwrite(str(out/'camera_first.png'),mat)
  else:
   finite=mat[np.isfinite(mat)&(mat>0)];first_images[kind]['valid_fraction']=float(len(finite)/mat.size)
   first_images[kind]['depth_percentiles']=np.percentile(finite,[1,50,99]).tolist() if len(finite) else []
   cv2.imwrite(str(out/'depth_first.png'),np.uint8(np.clip(np.nan_to_num(mat,posinf=12)/12,0,1)*255))
def clock_cb(msg):
 global clock_now
 clock_now=msg.clock.sec+msg.clock.nanosec*1e-9;clock.append([time.monotonic(),clock_now])
def info_cb(msg):
 if not camera_info:camera_info.update({'width':msg.width,'height':msg.height,'k':list(msg.k),'frame_id':msg.header.frame_id})
def odom_cb(msg):
 global yaw_previous,yaw_total,truth_pose
 q=msg.pose.pose.orientation;angle=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
 if yaw_previous is not None:yaw_total+=math.atan2(math.sin(angle-yaw_previous),math.cos(angle-yaw_previous))
 yaw_previous=angle;truth_pose=(msg.pose.pose.position.x,msg.pose.pose.position.y,angle)
 record('truth_odom',stamp(msg));v=msg.pose.pose;ow.writerow([stamp(msg),v.position.x,v.position.y,v.position.z,v.orientation.x,v.orientation.y,v.orientation.z,v.orientation.w])
def state_cb(msg):
 global tracking_first,current_state
 current_state=msg.data;states[msg.data]+=1
 if msg.data==2 and tracking_first is None:tracking_first=clock_now
subs=[node.create_subscription(Image,'/camera/color',lambda m:image(m,'rgb'),qos_profile_sensor_data),node.create_subscription(Image,'/camera/depth',lambda m:image(m,'depth'),qos_profile_sensor_data),node.create_subscription(CameraInfo,'/camera/info',info_cb,qos_profile_sensor_data),node.create_subscription(Clock,'/clock',clock_cb,qos_profile_sensor_data),node.create_subscription(Odometry,'/odom',odom_cb,qos_profile_sensor_data),node.create_subscription(Int32,'/slam/tracking_state',state_cb,10),node.create_subscription(Odometry,'/slam/odometry',lambda m:record('slam_odom',stamp(m)),10)]
subs.extend([node.create_subscription(PointCloud2,'/slam/map_points',map_cb,latched),node.create_subscription(RosPath,'/slam/path',lambda m:path_cb(m,'online_path'),latched),node.create_subscription(RosPath,'/slam/keyframes',lambda m:path_cb(m,'optimized_keyframes'),latched),node.create_subscription(TFMessage,'/tf',tf_cb,10),node.create_subscription(TFMessage,'/tf_static',tf_cb,latched)])
failure=None;route_finished=False
try:
 if a.record_bag:launch('bag',['ros2','bag','record','-o',str(out/'sensors'),'/camera/color','/camera/depth','/camera/info','/clock','/odom','/cmd_vel','/slam/tracking_state'])
 launch('gazebo',['ign','gazebo','-r','-s',*(['--headless-rendering'] if os.environ.get('X3_HEADLESS')=='1' else []),str(world),'-v','3'])
 launch('bridge',['ros2','run','ros_gz_bridge','parameter_bridge',
  '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
  '/x3/rgbd/image@sensor_msgs/msg/Image[ignition.msgs.Image',
  '/x3/rgbd/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
  '/x3/rgbd/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
  '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
  '/presentation/overview@sensor_msgs/msg/Image[ignition.msgs.Image',
  '/model/rosmaster_x3/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
  '--ros-args','-r','/x3/rgbd/image:=/camera/color','-r','/x3/rgbd/depth_image:=/camera/depth','-r','/x3/rgbd/camera_info:=/camera/info','-r','/model/rosmaster_x3/cmd_vel:=/cmd_vel'])
 while time.monotonic()-started<a.wall_limit:
  rclpy.spin_once(node,timeout_sec=0.02)
  if camera_info and len(first_images)==2 and slam_started is None:
   # Use measured intrinsics rather than assuming Gazebo's pixel-centre convention.
   text=(r/'config/x3_rgbd.yaml').read_text();import re
   for key,val in zip(['fx','fy','cx','cy'],[camera_info['k'][0],camera_info['k'][4],camera_info['k'][2],camera_info['k'][5]]):text=re.sub(r'Camera1\.'+key+r': .*',f'Camera1.{key}: {val}',text)
   text=re.sub(r'Camera.width: .*',f"Camera.width: {camera_info['width']}",text);text=re.sub(r'Camera.height: .*',f"Camera.height: {camera_info['height']}",text)
   text=re.sub(r'Camera.fps: .*',f'Camera.fps: {int(a.camera_fps)}',text)
   if a.route=='tour' or a.camera_height>.2:
    text=re.sub(r'Viewer.ViewpointY: .*','Viewer.ViewpointY: -10.0',text);text=re.sub(r'Viewer.ViewpointZ: .*','Viewer.ViewpointZ: -18.0',text)
   if a.save_atlas:text+='\nSystem.SaveAtlasToFile: "atlas"\n'
   if a.load_atlas:
    source=a.load_atlas.resolve()
    if not source.exists():raise RuntimeError('Atlas file does not exist: '+str(source))
    shutil.copy2(source,out/'loaded.osa');text+='\nSystem.LoadAtlasFromFile: "loaded"\n'
   (out/'camera.yaml').write_text(text)
   (out/'qos.json').write_text(json.dumps([{'node':x.node_name,'qos':str(x.qos_profile)} for x in node.get_publishers_info_by_topic('/camera/color')],indent=2))
   if not a.no_slam:launch('slam',[os.environ.get('X3_SLAM_BINARY',str(r/'build/runtime/x3_rgbd')),'--ros-args','-p','vocab:='+os.environ.get('X3_VOCAB','/workspace/x3-slam/Vocabulary/ORBvoc.txt'),'-p','use_sim_time:=true','-p','config:='+str(out/'camera.yaml'),'-p','output:='+str(out/'tracking.csv'),'-p','localization_map_id:='+str(a.localization_map_id),'-p','load_localization:='+str(bool(a.load_atlas)).lower(),'-p','diagnostics:='+str(a.diagnostics).lower(),'-p','viewer:='+str(a.viewer).lower(),'-p','camera_height:='+str(a.camera_height)])
   slam_started=clock_now;print('RGB-D live; intrinsics measured; SLAM starting. Output: '+str(out),flush=True)
  ready=(a.no_slam and slam_started is not None) or (states[2]>=5)
  if ready and route_start is None:route_start=clock_now;yaw_start=yaw_total;print('Starting bounded '+a.route+' route',flush=True)
  if clock_now-last_command>=0.1:
   cmd=Twist()
   if route_start is not None:
    dt=clock_now-route_start
    if a.route=='tour' and current_state!=2:
     if lost_since is None:lost_since=clock_now
     if clock_now-lost_since>8:failure='Tour stopped: tracking did not recover in 8 simulation seconds';break
    elif a.route=='tour' and truth_pose is not None:
     lost_since=None
     tx,ty,yaw=truth_pose;gx,gy=tour[goal_index];distance=math.hypot(gx-tx,gy-ty)
     if distance<.10:
      events.append({'event':'waypoint_reached','index':goal_index,'sim_time':clock_now,'position':[tx,ty]});goal_index+=1
      if goal_index>=len(tour):route_finished=True;break
     else:
      desired=math.atan2(gy-ty,gx-tx)
      if a.look_room:
       if tx<-2.5:target=(-6.,-1.) if ty<-1.7 else (-6.,2.)
       elif ty>3:target=(.5,4.5)
       elif tx>4.7:target=(6.5,1.)
       else:target=(.5,-2.)
       if math.hypot(target[0]-tx,target[1]-ty)>1.0:look_yaw=math.atan2(target[1]-ty,target[0]-tx)
       desired=look_yaw
      error=math.atan2(math.sin(desired-yaw),math.cos(desired-yaw))
      cmd.angular.z=max(-.30,min(.30,error*1.1))
      if abs(error)<.35:
       speed=min(a.speed,distance*.6)*max(0,math.cos(error));bearing=math.atan2(gy-ty,gx-tx)-yaw
       cmd.linear.x=speed*math.cos(bearing);cmd.linear.y=speed*math.sin(bearing) if a.look_room else 0.
    elif a.route=='forward':
     if 2<dt<10:cmd.linear.x=0.08
     elif 12<dt<20:cmd.linear.x=-0.08
    elif a.route=='rectangle':
     # Holonomic rectangle keeps the view forward; turns are tested separately.
     if 2<dt<22:cmd.linear.x=0.08
     elif 24<dt<36:cmd.linear.y=0.08
     elif 38<dt<58:cmd.linear.x=-0.08
     elif 60<dt<72:cmd.linear.y=-0.08
    elif a.route=='demo':
     if 2<dt<10:cmd.linear.x=0.08
     elif 12<dt<20:cmd.linear.x=-0.08
     elif dt>23 and yaw_total-yaw_start<4*math.pi:cmd.angular.z=0.30
     if yaw_total-yaw_start>=4*math.pi:route_finished=True;events.append({'event':'two_actual_rotations_completed','sim_time':clock_now,'yaw_radians':yaw_total-yaw_start});break
    elif a.route=='spin':
     if dt>2 and yaw_total-yaw_start<4*math.pi:cmd.angular.z=0.30
     if yaw_total-yaw_start>=4*math.pi:route_finished=True;events.append({'event':'two_actual_rotations_completed','sim_time':clock_now,'yaw_radians':yaw_total-yaw_start});break
    if a.recovery:
     if dt>=22 and not localization_on:
      modepub.publish(Bool(data=True));localization_on=True;events.append({'event':'localization_only_requested','sim_time':clock_now})
     wanted=24<=dt<28
     if wanted!=black_on:
      blackpub.publish(Bool(data=wanted));black_on=wanted;events.append({'event':'blackout_on' if wanted else 'blackout_off','sim_time':clock_now})
    if dt>a.sim_duration:
     route_finished=a.route not in ('spin','demo','tour')
     if not route_finished:failure='Spin time limit reached before two measured rotations'
     break
   commands.publish(cmd);last_command=clock_now
  for name,proc in processes:
   if proc.poll() is not None:raise RuntimeError(f'{name} exited early: {proc.returncode}')
  if slam_started is not None and route_start is None and clock_now-slam_started>15:raise RuntimeError('No stable initialization in 15 simulation seconds')
 if not first_images:failure='No camera images before wall-time limit'
 elif not route_finished and failure is None:failure='Wall-time limit reached before completing the route'
except (Exception,KeyboardInterrupt) as e:
 failure=str(e) or 'Interrupted';print(failure,flush=True)
finally:
 if rclpy.ok():
  for _ in range(3):commands.publish(Twist())
 # Stop SLAM first, allowing the worker to finish its last frame.
 exits={}
 for name,proc in reversed(processes):
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGINT)
   try:proc.wait(timeout=70 if name=='slam' else 6)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();events.append({'event':'forced_kill','process':name})
  exits[name]=proc.returncode
 stream_file.close();odom_file.close();node.destroy_node();rclpy.try_shutdown()
 for f in logs:f.close()
if any(code != 0 for code in exits.values()):
 failure = failure or 'Process exit failure: '+str(exits)
def rates(samples):
 if len(samples)<2:return {'messages':len(samples)}
 wall=samples[-1][0]-samples[0][0];sim=samples[-1][1]-samples[0][1]
 return {'messages':len(samples),'wall_duration_s':wall,'stamp_duration_s':sim,'wall_hz':(len(samples)-1)/wall if wall else None,'simulation_hz':(len(samples)-1)/sim if sim else None}
summary={'output':str(out),'mode':'live_headless_egl' if os.environ.get('X3_HEADLESS')=='1' else 'live_default_rendering','wall_limit_s':a.wall_limit,'elapsed_wall_s':time.monotonic()-started,'route_completed':route_finished,'failure':failure,'camera':first_images,'camera_info':camera_info,'streams':{k:rates(v) for k,v in streams.items()},'states':dict(states),'route_start_sim':route_start,'clock_final':clock_now}
summary.update({'events':events,'process_exits':exits,'save_requested':a.save_atlas,'load_requested':str(a.load_atlas) if a.load_atlas else None,'route':a.route,'measured_yaw_sweep_radians':yaw_total-yaw_start,'published_outputs':published_outputs,'speed_limit_m_s':a.speed,'waypoints_reached':goal_index-1 if tour else None,'waypoints_total':len(tour)-1 if tour else None})
if (out/'shutdown_complete.json').exists():summary['shutdown']=json.loads((out/'shutdown_complete.json').read_text())
if (out/'atlas.osa').exists():summary['saved_atlas_bytes']=(out/'atlas.osa').stat().st_size
if (out/'slam.log').exists():
 corelog=(out/'slam.log').read_text(errors='replace');summary['loop_corrections']=corelog.count('AUDIT_LOOP_CORRECTION_DONE');summary['relocalization_successes']=corelog.count('AUDIT_RELOCALIZED');summary['loaded_map_selected']='AUDIT_LOADED_MAP_SELECTED' in corelog
if route_start is not None:summary['route_streams']={k:rates([x for x in v if x[1]>=route_start]) for k,v in streams.items()}
if len(clock)>1:
 stable=[x for x in clock if x[1]>2]
 if len(stable)>1:summary['real_time_factor_measured']=(stable[-1][1]-stable[0][1])/(stable[-1][0]-stable[0][0])
if (out/'tracking.csv').exists():
 with (out/'tracking.csv').open() as f:rows=list(csv.DictReader(f))
 summary['tracking_frames']=len(rows)
 for field in ['process_ms','queue_ms','sim_age_s']:
  values=[float(x[field]) for x in rows];summary[field]={'p50':float(np.percentile(values,50)),'p95':float(np.percentile(values,95)),'max':max(values)} if values else {}
 if rows:summary['overwritten_pairs']=int(rows[-1]['overwritten_pairs']);summary['unpaired_discarded']=int(rows[-1]['unpaired_discarded'])
if a.load_atlas and (out/'atlas_loaded.csv').exists() and (out/'atlas_final.csv').exists():
 before=list(csv.DictReader((out/'atlas_loaded.csv').open()));after=list(csv.DictReader((out/'atlas_final.csv').open()))
 selected=next(int(x['map_id']) for x in before if x['selected']=='1')
 before_counts={int(x['map_id']):int(x['keyframes']) for x in before};after_counts={int(x['map_id']):int(x['keyframes']) for x in after}
 integrity={'selected_map':selected,'keyframes_before':before_counts,'keyframes_after':after_counts,'maps_and_keyframes_unchanged':before_counts==after_counts,'successful_relocalizations':summary.get('relocalization_successes',0)}
 integrity['passed']=integrity['maps_and_keyframes_unchanged'] and integrity['successful_relocalizations']>0 and summary.get('shutdown',{}).get('active_map')==selected
 summary['localization_integrity']=integrity
 if not integrity['passed']:summary['failure']=summary['failure'] or 'Loaded-map localization integrity check failed'
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
