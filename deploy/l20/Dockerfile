FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg2 locales software-properties-common && \
    locale-gen en_US.UTF-8 && \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
      > /etc/apt/sources.list.d/ros2.list && \
    curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
      -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main" \
      > /etc/apt/sources.list.d/gazebo-stable.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-ros-base ros-humble-cv-bridge ros-humble-image-transport \
      ros-humble-nav-msgs ros-humble-sensor-msgs ros-humble-std-msgs \
      ros-humble-tf2-ros ros-humble-ros-gz-bridge \
      ignition-fortress \
      build-essential cmake git pkg-config python3 python3-pip \
      libeigen3-dev libopencv-dev libboost-serialization-dev \
      libboost-system-dev libssl-dev libglew-dev libgl1 libegl1 \
      libglvnd0 libglx0 libx11-6 libxext6 libwayland-client0 \
      libjpeg-dev libpng-dev libepoxy-dev ffmpeg mesa-utils && \
    git clone --depth 1 --branch v0.8 \
      https://github.com/stevenlovegrove/Pangolin.git /tmp/Pangolin && \
    cmake -S /tmp/Pangolin -B /tmp/Pangolin/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
      -DBUILD_PANGOLIN_PYTHON=OFF && \
    cmake --build /tmp/Pangolin/build -j2 && \
    cmake --install /tmp/Pangolin/build && ldconfig && \
    rm -rf /tmp/Pangolin && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb openbox x11vnc novnc websockify xterm gdb xfonts-base fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=humble \
    PATH=/opt/ros/humble/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/usr/local/lib \
    PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages \
    AMENT_PREFIX_PATH=/opt/ros/humble \
    CMAKE_PREFIX_PATH=/opt/ros/humble

COPY container-entrypoint.sh /usr/local/bin/x3-entrypoint
RUN chmod 0755 /usr/local/bin/x3-entrypoint
ENTRYPOINT ["/usr/local/bin/x3-entrypoint"]


RUN apt-get update && apt-get install -y --no-install-recommends ros-humble-navigation2 ros-humble-nav2-bringup && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir gymnasium==1.2.2 && python3 -m pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
