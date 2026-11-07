import os
from pathlib import Path
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    root = Path(__file__).resolve().parent
    out = Path(os.environ['NAV_OUTPUT_DIR'])
    cfg = yaml.safe_load(Path('/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml').read_text())
    bt = cfg['bt_navigator']['ros__parameters']
    bt.update(global_frame='slam_map', robot_base_frame='slam_base_footprint',
              odom_topic='/nav/odometry', transform_tolerance=.3,
              default_nav_to_pose_bt_xml=str(root/'navigate.xml'),
              default_nav_through_poses_bt_xml=str(root/'navigate_through.xml'))
    ctrl = cfg['controller_server']['ros__parameters']
    ctrl.update(controller_frequency=10., failure_tolerance=2.)
    ctrl['progress_checker'].update(plugin='nav2_controller::PoseProgressChecker',
        required_movement_radius=.08, required_movement_angle=.05, movement_time_allowance=15.)
    ctrl['general_goal_checker'].update(xy_goal_tolerance=.10, yaw_goal_tolerance=.18)
    ctrl['FollowPath'] = dict(
        plugin='nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController',
        desired_linear_vel=.10, lookahead_dist=.30, min_lookahead_dist=.20,
        max_lookahead_dist=.40, lookahead_time=1.5, rotate_to_heading_angular_vel=.25,
        transform_tolerance=.3, use_velocity_scaled_lookahead_dist=False,
        min_approach_linear_velocity=.03, approach_velocity_scaling_dist=.4,
        use_collision_detection=True, max_allowed_time_to_collision_up_to_carrot=1.,
        use_regulated_linear_velocity_scaling=True, use_cost_regulated_linear_velocity_scaling=True,
        regulated_linear_scaling_min_radius=.5, regulated_linear_scaling_min_speed=.03,
        use_rotate_to_heading=True, rotate_to_heading_min_angle=.5,
        max_angular_accel=2.0, allow_reversing=False, cost_scaling_dist=.3,
        cost_scaling_gain=1., inflation_cost_scaling_factor=5.)
    for name in ['global_costmap', 'local_costmap']:
        p = cfg[name][name]['ros__parameters']
        p.update(global_frame='slam_map', robot_base_frame='slam_base_footprint',
                 update_frequency=3., publish_frequency=2., transform_tolerance=.3,
                 robot_radius=.19, track_unknown_space=True,
                 rolling_window=False, width=16, height=16,
                 plugins=['static_layer', 'inflation_layer'])
        p['static_layer'] = dict(plugin='nav2_costmap_2d::StaticLayer',
            map_topic='/map', map_subscribe_transient_local=True, subscribe_to_updates=False)
        p['inflation_layer'].update(inflation_radius=.34, cost_scaling_factor=5.)
    cfg['planner_server']['ros__parameters'].update(expected_planner_frequency=1.)
    cfg['planner_server']['ros__parameters']['GridBased'].update(
        tolerance=.05, use_astar=True, allow_unknown=False)
    file = out/'nav2_resolved.yaml'
    file.write_text(yaml.safe_dump(cfg))
    names = [('nav2_controller', 'controller_server'), ('nav2_planner', 'planner_server'),
             ('nav2_bt_navigator', 'bt_navigator')]
    nodes = [Node(package=pkg, executable=name, name=name, output='screen',
                  parameters=[str(file)], remappings=[('cmd_vel', '/nav/raw_cmd_vel'),
                                                     ('odom', '/nav/odometry')])
             for pkg, name in names]
    nodes.append(Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen', parameters=[dict(
            use_sim_time=True, autostart=True, node_names=[n for _, n in names],
            bond_timeout=4.)]))
    return LaunchDescription(nodes)
