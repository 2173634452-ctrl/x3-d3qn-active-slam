"""Start one fully isolated ROS/Gazebo episode using read-only X3 assets."""
import hashlib, json, os, shutil, sys
from pathlib import Path

source = Path('/workspace/x3-slam/scripts/benchmark_rgbd.py')
out = Path(sys.argv[sys.argv.index('--session-dir')+1])
out.mkdir(parents=True)
os.environ['NAV_OUTPUT_DIR'] = str(out)
root = out/'source'
root.mkdir()
for file in Path(__file__).resolve().parent.iterdir():
    if file.is_file() and (file.suffix in ('.py','.xml','.sh') or file.name == 'Dockerfile'):
        shutil.copy2(file,root/file.name)
(out/'source_sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest()
    for f in root.iterdir()},indent=2))
text = source.read_text()
if os.environ.get('X3_RECORD_VIDEO') == '1':
    import xml.etree.ElementTree as ET
    tree = ET.parse('/workspace/x3-slam/worlds/x3_house_rgbd.sdf')
    body = tree.find(".//model[@name='rosmaster_x3']/link[@name='base_footprint']")
    assert body is not None
    body.append(ET.fromstring('''<sensor name="x3_third_person" type="camera">
      <always_on>true</always_on><update_rate>10</update_rate>
      <topic>/x3/third_person</topic><pose>-1.2 -0.8 1.0 0 0.55 0.588</pose>
      <camera><horizontal_fov>1.15</horizontal_fov>
      <image><width>640</width><height>480</height><format>R8G8B8</format></image>
      <clip><near>0.05</near><far>20</far></clip></camera></sensor>'''))
    world = out/'recording_world.sdf'
    tree.write(world,encoding='unicode')
    sys.argv.extend(['--world',str(world)])
    old_bridge = "'/presentation/overview@sensor_msgs/msg/Image[ignition.msgs.Image',"
    assert text.count(old_bridge) == 1
    text = text.replace(old_bridge,old_bridge+"\n  '/x3/third_person@sensor_msgs/msg/Image[ignition.msgs.Image',")
assert text.count('commands.publish(cmd);last_command=clock_now') == 1
text = text.replace('commands.publish(cmd);last_command=clock_now','last_command=clock_now')
old = " launch('gazebo',['ign'"
assert text.count(old) == 1
text = text.replace(old,f""" launch('exploration_adapter',['python3','{root}/exploration_bridge.py','--ros-args','-p','use_sim_time:=true','-p','output_dir:='+str(out)])
 launch('nav2',['ros2','launch','{root}/nav_launch.py'])
 launch('episode_worker',['python3','{root}/episode_worker.py','--ros-args','-p','use_sim_time:=true','-p','output_dir:='+str(out)])
 {"launch('video_recorder',['python3','/workspace/x3-slam/learning/record_simulation.py','--ros-args','-p','use_sim_time:=true','-p','output_dir:='+str(out)])" if os.environ.get('X3_RECORD_VIDEO') == '1' else "# Recording disabled for this worker"}
 launch('gazebo',['ign'""")
old = '  for name,proc in processes:\n'
assert text.count(old) == 1
text = text.replace(old,"""  if (out/'episode_close.json').exists():route_finished=True;break
  if (out/'fatal.json').exists():failure=json.loads((out/'fatal.json').read_text())['error'];break
"""+old)
text = text.replace('out.mkdir(parents=True)','out.mkdir(parents=True,exist_ok=True)')
(out/'derived_benchmark.py').write_text(text)
exec(compile(text,str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
