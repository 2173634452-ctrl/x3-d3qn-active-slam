"""Read-only recorder of actual Gazebo cameras and estimated navigation map."""
import json,math,os,shutil,subprocess,time
from pathlib import Path
import cv2,numpy as np,rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid,Odometry
from geometry_msgs.msg import PoseStamped


class Recorder(Node):
    def __init__(self):
        super().__init__('x3_visual_recorder')
        self.declare_parameter('output_dir','/results/video')
        self.out=Path(self.get_parameter('output_dir').value)/'video'
        self.out.mkdir(parents=True,exist_ok=True)
        self.view=Path(os.environ['X3_VIEW_ROOT'])
        self.view.mkdir(parents=True,exist_ok=True)
        self.save_video=os.environ.get('X3_SAVE_VIDEO','0')=='1'
        self.bridge=CvBridge()
        self.writers={}
        self.last={}
        self.counts={}
        self.position=None
        self.goal=None
        self.create_subscription(Image,'/x3/third_person',lambda m:self.frame(m,'scene'),qos_profile_sensor_data)
        self.create_subscription(Image,'/camera/color',lambda m:self.frame(m,'front'),qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid,'/map',self.map,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Odometry,'/nav/odometry',self.odom,10)
        self.create_subscription(PoseStamped,'/exploration/goal',self.target,10)

    def odom(self,m):
        p=m.pose.pose.position
        self.position=(p.x,p.y)

    def target(self,m):
        self.goal=(m.pose.position.x,m.pose.position.y)

    def jpeg(self,path,mat):
        ok,data=cv2.imencode('.jpg',mat,[cv2.IMWRITE_JPEG_QUALITY,85])
        if ok:
            tmp=path.with_suffix('.tmp')
            tmp.write_bytes(data.tobytes())
            tmp.replace(path)

    def frame(self,m,kind):
        stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        if stamp-self.last.get(kind,-100)<.19:
            return
        self.last[kind]=stamp
        mat=self.bridge.imgmsg_to_cv2(m,'bgr8')
        if kind not in self.counts:
            self.counts[kind]=0
            self.jpeg(self.out/(kind+'_first.jpg'),mat)
        if self.save_video and kind not in self.writers:
            cmd=['ffmpeg','-nostdin','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24',
                '-s',f'{m.width}x{m.height}','-r','5','-i','pipe:0','-an','-c:v','libx264',
                '-preset','ultrafast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(self.out/(kind+'.mp4'))]
            self.writers[kind]=subprocess.Popen(cmd,stdin=subprocess.PIPE,start_new_session=True)
        if self.save_video:
            try:
                self.writers[kind].stdin.write(np.ascontiguousarray(mat).tobytes())
            except BrokenPipeError:
                self.get_logger().error('Video encoder stopped: '+kind)
        self.counts[kind]+=1
        self.jpeg(self.view/(kind+'.jpg'),mat)
        info=dict(sim_time=stamp,frames=self.counts,episode=self.out.parent.name,wall=time.time())
        tmp=self.view/'camera.tmp'
        tmp.write_text(json.dumps(info))
        tmp.replace(self.view/'camera.json')

    def map(self,m):
        a=np.array(m.data).reshape(m.info.height,m.info.width)
        gray=np.where(a<0,125,np.where(a==0,245,25)).astype(np.uint8)
        mat=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        def cell(xy):
            return (int((xy[0]-m.info.origin.position.x)/m.info.resolution),
                    int((xy[1]-m.info.origin.position.y)/m.info.resolution))
        if self.position:
            cv2.circle(mat,cell(self.position),4,(40,90,240),-1)
        if self.goal:
            cv2.drawMarker(mat,cell(self.goal),(40,200,40),cv2.MARKER_CROSS,10,2)
        self.jpeg(self.view/'map.jpg',cv2.flip(mat,0))

    def close(self):
        for kind,proc in self.writers.items():
            try:
                proc.stdin.close()
                code=proc.wait(timeout=20)
                if code==0:
                    tmp=self.view/(kind+'_latest.tmp')
                    shutil.copyfile(self.out/(kind+'.mp4'),tmp)
                    tmp.replace(self.view/(kind+'_latest.mp4'))
            except Exception:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        (self.out/'recording.json').write_text(json.dumps(dict(frames=self.counts,fps=5)))


if __name__=='__main__':
    rclpy.init()
    node=Recorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
