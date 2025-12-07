---
id: chapter-02-02-gazebo-basics
title: "Gazebo Basics"
sidebar_label: "Chapter 2: Gazebo Basics"
description: "Mastering Gazebo simulation environment setup and configuration for robot models"
keywords:
  - Gazebo
  - Simulation
  - Physics
  - Robot Models
prerequisites:
  - chapter-01-01-architecture
  - chapter-01-02-nodes-topics-services
  - chapter-02-01-simulation-fundamentals
---


# Chapter 2 - Gazebo Basics

## Learning Objectives

- Master Gazebo simulation environment setup and configuration
- Create and configure robot models for Gazebo simulation
- Implement sensor integration in simulated environments
- Configure physics properties and collision detection
- Debug and optimize simulation performance

## Prerequisites

- Understanding of physics simulation fundamentals
- Completed Chapter 1: Physics Simulation Fundamentals
- Ubuntu 22.04 LTS with ROS 2 Humble and Gazebo Garden installed
- Basic knowledge of URDF robot modeling
- Familiarity with ROS 2 communication patterns

## Introduction

Gazebo is a powerful open-source physics simulator widely used in robotics research and development. It provides realistic simulation of robots in complex environments with accurate physics, sensors, and rendering capabilities. This chapter will guide you through advanced Gazebo concepts and practical implementations for robotic simulation.

Gazebo's strength lies in its ability to simulate complex multi-robot scenarios with realistic physics, making it an essential tool for developing and testing robotic systems before deployment to physical hardware.

### Why Gazebo for Robotics

Gazebo offers several advantages for robotic simulation:
- Accurate physics simulation with multiple physics engines
- Extensive sensor models (cameras, lidar, IMU, etc.)
- Flexible environment creation and modification
- Strong integration with ROS 2 ecosystem
- Active community and extensive documentation

### Real-world Applications

- Autonomous vehicle testing in urban environments
- Drone swarm coordination and collision avoidance
- Industrial robot programming and validation
- Human-robot interaction research

### What You'll Build by the End

By completing this chapter, you will create:
- A complete robot model with multiple sensors in Gazebo
- Custom world environments with complex objects
- Sensor data processing pipelines
- Performance optimization techniques for simulation

## Core Concepts

### Gazebo Architecture

Gazebo consists of several key components:
- **Gazebo Server**: Core simulation engine that handles physics and rendering
- **Gazebo Client**: Visualization interface for human interaction
- **Gazebo Plugins**: Extensions that provide ROS 2 integration and custom functionality
- **Model Database**: Repository of pre-built models and environments

### Physics Simulation Parameters

Gazebo's physics simulation can be tuned with various parameters:
- Gravity settings
- Solver algorithms and parameters
- Collision detection algorithms
- Real-time update rates

### Sensor Integration

Gazebo provides realistic sensor simulation including:
- Camera sensors with configurable parameters
- 3D lidar and 2D laser range finders
- IMU and accelerometer sensors
- Force/torque sensors
- GPS and magnetometer simulation

## Hands-On Tutorial

### Step 1: Create a Detailed Robot Model with Sensors

First, let's create a more complex robot model with integrated sensors:

```bash
mkdir -p ~/gazebo_ws/src/detailed_robot_description/urdf
cd ~/gazebo_ws/src/detailed_robot_description
```

Create `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>detailed_robot_description</name>
  <version>0.0.0</version>
  <description>Detailed robot model with sensors for Gazebo</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(detailed_robot_description)

find_package(ament_cmake REQUIRED)

# Install launch files
install(DIRECTORY
  urdf
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

Create `urdf/detailed_robot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="detailed_robot">
  <!-- Constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Base properties -->
  <xacro:property name="base_mass" value="10.0" />
  <xacro:property name="base_length" value="0.8" />
  <xacro:property name="base_width" value="0.6" />
  <xacro:property name="base_height" value="0.3" />

  <!-- Wheel properties -->
  <xacro:property name="wheel_radius" value="0.15" />
  <xacro:property name="wheel_width" value="0.05" />
  <xacro:property name="wheel_mass" value="0.5" />

  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 1.0 1.0"/>
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="red">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${base_mass}"/>
      <inertia
        ixx="${base_mass/12.0 * (base_width*base_width + base_height*base_height)}"
        ixy="0.0"
        ixz="0.0"
        iyy="${base_mass/12.0 * (base_length*base_length + base_height*base_height)}"
        iyz="0.0"
        izz="${base_mass/12.0 * (base_length*base_length + base_width*base_width)}"/>
    </inertial>
  </link>

  <!-- Gazebo plugins configuration -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <!-- Macro for wheels -->
  <xacro:macro name="wheel" params="prefix *joint_pose">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${wheel_mass}"/>
        <inertia
          ixx="${wheel_mass/12.0 * (3*wheel_radius*wheel_radius + wheel_width*wheel_width)}"
          ixy="0.0"
          ixz="0.0"
          iyy="${wheel_mass/12.0 * (3*wheel_radius*wheel_radius + wheel_width*wheel_width)}"
          iyz="0.0"
          izz="${wheel_mass * wheel_radius * wheel_radius}"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <xacro:insert_block name="joint_pose"/>
      <axis xyz="0 1 0"/>
    </joint>

    <!-- Gazebo wheel plugin -->
    <gazebo reference="${prefix}_wheel">
      <material>Gazebo/Black</material>
      <mu1>0.8</mu1>
      <mu2>0.8</mu2>
      <kp>1000000.0</kp>
      <kd>100.0</kd>
    </gazebo>
  </xacro:macro>

  <!-- Define wheels -->
  <xacro:wheel prefix="front_left">
    <origin xyz="${base_length/2} ${base_width/2 - wheel_width/2} 0" rpy="0 0 0"/>
  </xacro:wheel>

  <xacro:wheel prefix="front_right">
    <origin xyz="${base_length/2} ${-base_width/2 + wheel_width/2} 0" rpy="0 0 0"/>
  </xacro:wheel>

  <xacro:wheel prefix="rear_left">
    <origin xyz="${-base_length/2} ${base_width/2 - wheel_width/2} 0" rpy="0 0 0"/>
  </xacro:wheel>

  <xacro:wheel prefix="rear_right">
    <origin xyz="${-base_length/2} ${-base_width/2 + wheel_width/2} 0" rpy="0 0 0"/>
  </xacro:wheel>

  <!-- Camera sensor -->
  <joint name="camera_joint" type="fixed">
    <origin xyz="${base_length/2 - 0.05} 0 ${base_height/2}" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="camera_link"/>
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="1e-6" ixy="0.0" ixz="0.0" iyy="1e-6" iyz="0.0" izz="1e-6"/>
    </inertial>
  </link>

  <!-- Gazebo camera plugin -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera1">
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>800</width>
          <height>600</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin filename="gz-sim-camera-system" name="gz::sim::systems::Camera">
        <update_rate>30</update_rate>
        <topic_name>/detailed_robot/camera/image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- 3D Lidar sensor -->
  <joint name="lidar_joint" type="fixed">
    <origin xyz="0 0 ${base_height/2 + 0.1}" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="lidar_link"/>
  </joint>

  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="1e-5" ixy="0.0" ixz="0.0" iyy="1e-5" iyz="0.0" izz="1e-5"/>
    </inertial>
  </link>

  <!-- Gazebo lidar plugin -->
  <gazebo reference="lidar_link">
    <sensor type="gpu_lidar" name="gpu_lidar">
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>30</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
        <topic_name>/detailed_robot/lidar/scan</topic_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU sensor -->
  <joint name="imu_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="imu_link"/>
  </joint>

  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="1e-6" ixy="0.0" ixz="0.0" iyy="1e-6" iyz="0.0" izz="1e-6"/>
    </inertial>
  </link>

  <!-- Gazebo IMU plugin -->
  <gazebo reference="imu_link">
    <sensor type="imu" name="imu_sensor">
      <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu">
        <topic_name>/detailed_robot/imu/data</topic_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Differential drive plugin -->
  <gazebo>
    <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
      <left_joint>front_left_wheel_joint</left_joint>
      <right_joint>front_right_wheel_joint</right_joint>
      <wheel_separation>0.6</wheel_separation>
      <wheel_radius>0.15</wheel_radius>
      <odom_publish_frequency>30</odom_publish_frequency>
      <topic_name>/detailed_robot/cmd_vel</topic_name>
      <odom_topic_name>/detailed_robot/odom</odom_topic_name>
      <tf_topic_name>/detailed_robot/tf</tf_topic_name>
    </plugin>
  </gazebo>
</robot>
```

### Step 2: Create a Gazebo World File

Create a custom world for our simulation:

```bash
mkdir -p ~/gazebo_ws/src/detailed_robot_simulation/worlds
cd ~/gazebo_ws/src/detailed_robot_simulation
```

Create `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>detailed_robot_simulation</name>
  <version>0.0.0</version>
  <description>Detailed robot simulation package</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>launch</depend>
  <depend>launch_ros</depend>
  <depend>detailed_robot_description</depend>
  <depend>ros_gz_sim</depend>
  <depend>ros_gz_bridge</depend>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(detailed_robot_simulation)

find_package(ament_cmake REQUIRED)
find_package(launch REQUIRED)
find_package(launch_ros REQUIRED)
find_package(ros_gz_sim REQUIRED)
find_package(ros_gz_bridge REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  worlds
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

Create `worlds/robotics_lab.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="robotics_lab">
    <!-- Include the outdoor environment -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Laboratory walls -->
    <model name="wall_1">
      <pose>0 5 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="wall_2">
      <pose>0 -5 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="wall_3">
      <pose>10 0 1 0 0 1.5707</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="wall_4">
      <pose>-10 0 1 0 0 1.5707</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Add some obstacles -->
    <model name="obstacle_1">
      <pose>-2 2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.4 1.0 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="obstacle_2">
      <pose>3 -1 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>1</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>1.0 0.4 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
    </model>

    <!-- Add a table for testing -->
    <model name="table">
      <pose>0 0 0.4 0 0 0</pose>
      <link name="table_top">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1 0.05</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1 0.05</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>20</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
      <link name="leg_1">
        <pose>-0.9 -0.45 0 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.2 0.1 1</ambient>
            <diffuse>0.5 0.4 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
      <link name="leg_2">
        <pose>0.9 -0.45 0 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.2 0.1 1</ambient>
            <diffuse>0.5 0.4 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
      <link name="leg_3">
        <pose>-0.9 0.45 0 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.2 0.1 1</ambient>
            <diffuse>0.5 0.4 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
      <link name="leg_4">
        <pose>0.9 0.45 0 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.2 0.1 1</ambient>
            <diffuse>0.5 0.4 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertial>
        </inertial>
      </link>
      <joint name="top_to_leg1" type="fixed">
        <parent>table_top</parent>
        <child>leg_1</child>
      </joint>
      <joint name="top_to_leg2" type="fixed">
        <parent>table_top</parent>
        <child>leg_2</child>
      </joint>
      <joint name="top_to_leg3" type="fixed">
        <parent>table_top</parent>
        <child>leg_3</child>
      </joint>
      <joint name="top_to_leg4" type="fixed">
        <parent>table_top</parent>
        <child>leg_4</child>
      </joint>
    </model>
  </world>
</sdf>
```

### Step 3: Create Simulation Launch Files

Create `launch/detailed_robot_gazebo.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    world = LaunchConfiguration('world')
    robot_name = LaunchConfiguration('robot_name')

    # Include Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': ['-r -v 3', PathJoinSubstitution([
                FindPackageShare('detailed_robot_simulation'),
                'worlds',
                'robotics_lab.sdf'
            ])]
        }.items()
    )

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': open(PathJoinSubstitution([
                FindPackageShare('detailed_robot_description'),
                'urdf',
                'detailed_robot.urdf.xacro'
            ]).perform({})).read()
        }]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'detailed_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5',
            '-allow_renaming', 'true'
        ],
        output='screen'
    )

    # Bridge for sensor data
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/detailed_robot/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/detailed_robot/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/detailed_robot/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/detailed_robot/lidar/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/detailed_robot/imu/data@sensor_msgs/msg/Imu@gz.msgs.IMU',
            '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            '/clock@builtin_interfaces/msg/Time@gz.msgs.Clock'
        ],
        output='screen'
    )

    # RViz for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('detailed_robot_simulation'),
            'config',
            'detailed_robot.rviz'
        ])],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='robotics_lab.sdf',
            description='Choose one of the world files from `/detailed_robot_simulation/worlds`'
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value='detailed_robot',
            description='Name of the robot'
        ),
        gazebo,
        robot_state_publisher,
        spawn_entity,
        bridge,
        # rviz  # Comment out RViz for now to reduce complexity
    ])
```

### Step 4: Create Sensor Processing Nodes

Create a directory for our sensor processing nodes:

```bash
mkdir -p ~/gazebo_ws/src/detailed_robot_simulation/detailed_robot_simulation
```

Create `detailed_robot_simulation/sensor_processor.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import numpy as np
from cv_bridge import CvBridge
import cv2

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/detailed_robot/cmd_vel', 10)
        self.obstacle_distance_pub = self.create_publisher(Float32, 'obstacle_distance', 10)

        # Create subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan, '/detailed_robot/lidar/scan', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/detailed_robot/imu/data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/detailed_robot/odom', self.odom_callback, 10)

        # Initialize CvBridge for image processing
        self.bridge = CvBridge()

        # State variables
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.min_distance = float('inf')

        # Timer for sending commands
        self.timer = self.create_timer(0.1, self.send_commands)

        self.get_logger().info('Sensor Processor initialized')

    def lidar_callback(self, msg):
        """Process lidar data to detect obstacles"""
        # Find minimum distance in front of robot (±30 degrees)
        front_ranges = msg.ranges[150:210]  # Assuming 360 degree scan
        front_ranges = [r for r in front_ranges if r > msg.range_min and r < msg.range_max]

        if front_ranges:
            self.min_distance = min(front_ranges)
        else:
            self.min_distance = float('inf')

        # Publish obstacle distance
        dist_msg = Float32()
        dist_msg.data = self.min_distance
        self.obstacle_distance_pub.publish(dist_msg)

        self.get_logger().info(f'Min front distance: {self.min_distance:.2f}m')

    def imu_callback(self, msg):
        """Process IMU data"""
        # Extract orientation and angular velocity
        orientation = msg.orientation
        angular_velocity = msg.angular_velocity

        # In a real system, we'd use this for navigation and stability
        self.get_logger().info(f'IMU: Orientation Z={orientation.z:.2f}, Angular Vel Z={angular_velocity.z:.2f}')

    def odom_callback(self, msg):
        """Process odometry data"""
        # Extract position and velocity
        position = msg.pose.pose.position
        velocity = msg.twist.twist.linear

        self.get_logger().info(f'Odom: Pos=({position.x:.2f}, {position.y:.2f}), Vel=({velocity.x:.2f}, {velocity.y:.2f})')

    def send_commands(self):
        """Send movement commands based on sensor data"""
        cmd = Twist()

        # Simple obstacle avoidance algorithm
        if self.min_distance < 1.0:  # Too close to obstacle
            cmd.linear.x = 0.0  # Stop
            cmd.angular.z = 0.5  # Turn
        elif self.min_distance < 2.0:  # Getting close
            cmd.linear.x = 0.3  # Slow down
            cmd.angular.z = 0.0
        else:  # Clear path
            cmd.linear.x = 0.5  # Move forward
            cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = SensorProcessor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Update the CMakeLists.txt to include the new Python node:

```cmake
cmake_minimum_required(VERSION 3.8)
project(detailed_robot_simulation)

find_package(ament_cmake REQUIRED)
find_package(launch REQUIRED)
find_package(launch_ros REQUIRED)
find_package(ros_gz_sim REQUIRED)
find_package(ros_gz_bridge REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  worlds
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  detailed_robot_simulation/sensor_processor.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
```

### Step 5: Build and Test the Simulation

```bash
cd ~/gazebo_ws

# Source ROS 2 and Gazebo
source /opt/ros/humble/setup.bash
source /usr/share/gz/garden/setup.sh

# Build the packages
colcon build --packages-select detailed_robot_description detailed_robot_simulation

# Source the workspace
source install/setup.bash
```

Test the simulation:

```bash
# Launch the detailed robot simulation
source ~/gazebo_ws/install/setup.bash
ros2 launch detailed_robot_simulation detailed_robot_gazebo.launch.py
```

In another terminal, run the sensor processor:

```bash
# Terminal 2: Run sensor processor
source ~/gazebo_ws/install/setup.bash
ros2 run detailed_robot_simulation sensor_processor.py
```

Expected output: The robot should navigate around the obstacles in the custom world while processing sensor data.

### Common Issues

- **Plugin loading errors**: Ensure all required Gazebo plugins are available
- **Sensor data not publishing**: Check bridge configuration and topic names
- **Physics simulation instability**: Adjust physics parameters in the world file

## Practical Example

Let's create a complete autonomous navigation simulation:

```python
# ~/gazebo_ws/src/detailed_robot_simulation/detailed_robot_simulation/navigation_demo.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import math
import numpy as np

class NavigationDemo(Node):
    def __init__(self):
        super().__init__('navigation_demo')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/detailed_robot/cmd_vel', 10)
        self.goal_reached_pub = self.create_publisher(Float32, 'goal_reached', 10)

        self.lidar_sub = self.create_subscription(
            LaserScan, '/detailed_robot/lidar/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/detailed_robot/odom', self.odom_callback, 10)

        # Navigation parameters
        self.current_position = Point()
        self.current_orientation = 0.0
        self.lidar_data = None
        self.goal = Point()
        self.goal.x = 5.0
        self.goal.y = 5.0

        # Navigation state
        self.navigation_state = 'exploring'  # exploring, navigating_to_goal, avoiding_obstacles
        self.obstacle_detected = False
        self.safe_distance = 1.0

        # Timer for navigation control
        self.timer = self.create_timer(0.1, self.navigation_control)

        self.get_logger().info(f'Navigation Demo started, goal: ({self.goal.x}, {self.goal.y})')

    def lidar_callback(self, msg):
        """Process lidar data for obstacle detection"""
        self.lidar_data = msg.ranges
        min_distance = min([r for r in msg.ranges if r > msg.range_min and r < msg.range_max])

        self.obstacle_detected = min_distance < self.safe_distance

    def odom_callback(self, msg):
        """Update robot position and orientation"""
        self.current_position.x = msg.pose.pose.position.x
        self.current_position.y = msg.pose.pose.position.y

        # Extract orientation from quaternion
        quat = msg.pose.pose.orientation
        self.current_orientation = math.atan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                                              1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z))

    def calculate_distance_to_goal(self):
        """Calculate Euclidean distance to goal"""
        dx = self.goal.x - self.current_position.x
        dy = self.goal.y - self.current_position.y
        return math.sqrt(dx*dx + dy*dy)

    def calculate_angle_to_goal(self):
        """Calculate angle to goal relative to current orientation"""
        dx = self.goal.x - self.current_position.x
        dy = self.goal.y - self.current_position.y
        goal_angle = math.atan2(dy, dx)
        return goal_angle - self.current_orientation

    def navigation_control(self):
        """Main navigation control logic"""
        if not self.lidar_data:
            return

        cmd = Twist()
        distance_to_goal = self.calculate_distance_to_goal()

        # Check if goal is reached
        if distance_to_goal < 0.5:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info('Goal reached!')

            # Publish goal reached message
            goal_msg = Float32()
            goal_msg.data = 1.0
            self.goal_reached_pub.publish(goal_msg)
            return

        # Obstacle avoidance
        if self.obstacle_detected:
            self.get_logger().info('Obstacle detected, executing avoidance maneuver')
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn right
        else:
            # Navigate toward goal
            angle_to_goal = self.calculate_angle_to_goal()

            # Normalize angle to [-pi, pi]
            while angle_to_goal > math.pi:
                angle_to_goal -= 2 * math.pi
            while angle_to_goal < -math.pi:
                angle_to_goal += 2 * math.pi

            # Set velocities based on angle to goal
            cmd.linear.x = 0.5  # Forward speed
            cmd.angular.z = max(-0.5, min(0.5, angle_to_goal * 0.5))  # Proportional turning

        self.cmd_vel_pub.publish(cmd)

        self.get_logger().info(f'Pos: ({self.current_position.x:.2f}, {self.current_position.y:.2f}), '
                              f'Dist to goal: {distance_to_goal:.2f}, Angle: {math.degrees(angle_to_goal):.2f}°')

def main(args=None):
    rclpy.init(args=args)
    node = NavigationDemo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Add this to the CMakeLists.txt:

```cmake
# Install Python executables
install(PROGRAMS
  detailed_robot_simulation/sensor_processor.py
  detailed_robot_simulation/navigation_demo.py
  DESTINATION lib/${PROJECT_NAME}
)
```

Rebuild and test the navigation demo:

```bash
cd ~/gazebo_ws

# Source ROS 2 and Gazebo
source /opt/ros/humble/setup.bash
source /usr/share/gz/garden/setup.sh

# Rebuild the package
colcon build --packages-select detailed_robot_simulation

# Source the workspace
source install/setup.bash

# Launch the simulation
ros2 launch detailed_robot_simulation detailed_robot_gazebo.launch.py
```

In another terminal:

```bash
# Run the navigation demo
source ~/gazebo_ws/install/setup.bash
ros2 run detailed_robot_simulation navigation_demo.py
```

Expected results: The robot should navigate autonomously to the goal position while avoiding obstacles in the simulated environment.

## Troubleshooting

### Common Error 1: Sensor Bridge Issues
**Cause**: Incorrect topic mappings between Gazebo and ROS 2
**Solution**: Verify bridge configuration and topic names match
**Prevention Tips**: Use consistent naming conventions across all components

### Common Error 2: Physics Simulation Instability
**Cause**: Inappropriate physics parameters or unrealistic inertial properties
**Solution**: Adjust solver parameters and verify URDF inertial values
**Prevention Tips**: Use proper CAD models for accurate inertial calculations

### Common Error 3: Performance Issues
**Cause**: Complex models or high update rates overwhelming the system
**Solution**: Optimize model complexity and adjust simulation parameters
**Prevention Tips**: Profile simulation performance during development

## Key Takeaways

- Gazebo provides realistic physics simulation for robotic development
- Proper sensor integration is crucial for effective simulation
- Physics parameters significantly impact simulation accuracy
- Collision detection and response require careful tuning
- Performance optimization is essential for real-time simulation

## Additional Resources

- [Gazebo Tutorials](https://gazebosim.org/tutorials)
- [ROS 2 Gazebo Integration Guide](https://github.com/gazebosim/ros_gz/tree/humble)
- [URDF and Gazebo Integration](http://gazebosim.org/tutorials?tut=ros2_overview)
- [Sensor Configuration in Gazebo](https://gazebosim.org/api/sim/7/sensors.html)

## Self-Assessment

1. How do you configure a camera sensor in a Gazebo robot model?
2. What are the key physics parameters that affect simulation accuracy?
3. How do you bridge Gazebo topics to ROS 2 topics?
4. What are the differences between CPU and GPU-based sensors in Gazebo?
5. How would you optimize a Gazebo simulation for better performance?

<ChapterNavigation
  previous={{
    permalink: '/docs/module-02-digital-twin/chapter-02-01-simulation-fundamentals',
    title: '2.1 Simulation Fundamentals'
  }}
  next={{
    permalink: '/docs/module-02-digital-twin/chapter-02-03-unity-visualization',
    title: '2.3 Unity Visualization'
  }}
/>