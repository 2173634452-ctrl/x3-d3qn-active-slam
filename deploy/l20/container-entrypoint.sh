#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-184}"
export IGN_PARTITION="${IGN_PARTITION:-x3_aslam}"

exec "$@"
