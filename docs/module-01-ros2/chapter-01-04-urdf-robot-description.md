---
id: chapter-01-04-urdf-robot-description
title: "Chapter 1.4: URDF and Robot Description"
sidebar_label: "1.4 URDF & Robot Description"
sidebar_position: 4
description: "Master URDF (Unified Robot Description Format) for humanoid robots, including joint definitions, kinematics, collision models, and ROS 2 integration."
keywords: [URDF, robot description, humanoid kinematics, ROS 2, xacro, robot_state_publisher]
---

# Chapter 1.4: URDF and Robot Description

## Learning Objectives

By the end of this chapter, you will be able to:

- **Understand** the URDF format and its role in robot modeling
- **Create** URDF models for humanoid robots with complex kinematic chains
- **Define** joints, links, visual and collision geometries
- **Integrate** sensors and actuators into URDF descriptions
- **Use** Xacro for parameterized and modular robot descriptions
- **Visualize** and debug URDF models using RViz 2
- **Integrate** URDF with ROS 2's robot_state_publisher

## Prerequisites

- Understanding of ROS 2 architecture (Chapter 1.1)
- Familiarity with nodes, topics, and services (Chapter 1.2)
- Knowledge of ROS 2 workspaces and packages (Chapter 1.3)
- Basic understanding of 3D coordinate systems and transformations
- Familiarity with XML syntax

---

## 1.4.1 Understanding URDF

### What is URDF?

The **Unified Robot Description Format (URDF)** is an XML-based standard for representing the physical structure of a robot in ROS. URDF files describe:

- **Links**: Rigid bodies that form the robot's structure
- **Joints**: Connections between links that allow motion
- **Visual Models**: 3D meshes or geometric primitives for visualization
- **Collision Models**: Simplified geometries for collision detection
- **Inertial Properties**: Mass, center of mass, and moment of inertia
- **Sensors and Actuators**: Cameras, LiDAR, IMU, etc.

### Why URDF for Humanoid Robots?

Humanoid robots present unique challenges:

- **Complex Kinematic Chains**: 20+ degrees of freedom (arms, legs, torso, head)
- **Balance Requirements**: Accurate mass distribution and inertial properties
- **Sensor Integration**: Multiple cameras, IMUs, force-torque sensors
- **Dual-Arm Manipulation**: Coordinated control of both arms
- **Bipedal Locomotion**: Legs with 6+ DOF each

URDF provides a standardized way to model these complexities.

### URDF vs. Other Formats

| Format | Use Case | Strengths | Limitations |
|--------|----------|-----------|-------------|
| **URDF** | ROS robot modeling | ROS integration, widely supported | No closed kinematic loops |
| **SDF** | Gazebo simulation | Supports multiple robots, plugins | Less ROS tooling |
| **MuJoCo XML** | Physics simulation | Fast physics, advanced contact models | Limited ROS integration |
| **MJCF** | Reinforcement learning | Efficient for ML training | Not ROS-native |

---

## 1.4.2 Basic URDF Structure

### Minimal Humanoid URDF Example

Here's a simplified humanoid torso with one arm:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">

  <!-- Base Link (Torso) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.4" ixy="0.0" ixz="0.0"
               iyy="0.5" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Shoulder Link -->
  <link name="shoulder_link">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0"
               iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Shoulder Joint (Revolute) -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0.0 0.15 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
  </joint>

  <!-- Upper Arm Link -->
  <link name="upper_arm_link">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.008" ixy="0.0" ixz="0.0"
               iyy="0.008" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Elbow Joint -->
  <joint name="elbow_joint" type="revolute">
    <parent link="shoulder_link"/>
    <child link="upper_arm_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="50" velocity="2.0"/>
  </joint>

</robot>
```

### Key URDF Elements

#### Links

Links represent rigid bodies:

```xml
<link name="link_name">
  <visual>      <!-- What you see in visualization -->
  <collision>   <!-- What's used for collision detection -->
  <inertial>    <!-- Physical properties for dynamics -->
</link>
```

#### Joints

Joints connect links and define motion constraints:

**Joint Types for Humanoids:**

| Joint Type | Description | Humanoid Use Case |
|------------|-------------|-------------------|
| `revolute` | Rotation with limits | Elbows, knees, hips |
| `continuous` | Unlimited rotation | Head pan (rare) |
| `prismatic` | Linear motion | Telescoping spine (rare) |
| `fixed` | No motion | Rigidly attached sensors |
| `floating` | 6-DOF free motion | Base link in simulation |
| `planar` | 2D planar motion | Not common in humanoids |

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>  <!-- Position relative to parent -->
  <axis xyz="0 0 1"/>                 <!-- Rotation axis -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
  <dynamics damping="0.7" friction="1.0"/>
</joint>
```

#### Coordinate Systems and Transforms

URDF uses **right-handed coordinate systems** with the convention:
- **X**: Forward (red)
- **Y**: Left (green)
- **Z**: Up (blue)

**Roll-Pitch-Yaw (RPY)** angles are specified in radians:
```xml
<origin xyz="x y z" rpy="roll pitch yaw"/>
```

---

## 1.4.3 Humanoid Robot URDF Structure

### Full Humanoid Skeleton

A typical humanoid has these major components:

```
base_link (torso)
├── head_link
│   ├── neck_pitch_joint
│   └── neck_yaw_joint
├── left_arm
│   ├── left_shoulder_pitch_joint
│   ├── left_shoulder_roll_joint
│   ├── left_shoulder_yaw_joint
│   ├── left_elbow_pitch_joint
│   ├── left_wrist_roll_joint
│   └── left_hand_link
├── right_arm (mirror of left)
├── left_leg
│   ├── left_hip_yaw_joint
│   ├── left_hip_roll_joint
│   ├── left_hip_pitch_joint
│   ├── left_knee_pitch_joint
│   ├── left_ankle_pitch_joint
│   ├── left_ankle_roll_joint
│   └── left_foot_link
└── right_leg (mirror of left)
```

### Degrees of Freedom Breakdown

| Body Part | Typical DOF | Joints |
|-----------|-------------|--------|
| **Head** | 2-3 | Neck pitch, yaw, (roll) |
| **Torso** | 0-3 | Fixed or waist pitch/roll/yaw |
| **Arm (each)** | 6-7 | Shoulder (3), elbow (1-2), wrist (2) |
| **Leg (each)** | 6 | Hip (3), knee (1), ankle (2) |
| **Hand (each)** | 0-20 | Fingers (optional) |
| **Total** | 20-50+ | Full humanoid |

---

## 1.4.4 Using Xacro for Modular URDF

### What is Xacro?

**Xacro** (XML Macros) extends URDF with:
- **Macros**: Reusable components
- **Properties**: Constants and variables
- **Math**: Calculations within the file
- **Conditional Logic**: Include/exclude components

### Converting URDF to Xacro

**Basic Xacro File Structure:**

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid">

  <!-- Properties (Constants) -->
  <xacro:property name="arm_length" value="0.3"/>
  <xacro:property name="arm_radius" value="0.04"/>
  <xacro:property name="arm_mass" value="1.0"/>

  <!-- Math Calculations -->
  <xacro:property name="arm_ixx" value="${arm_mass * (3*arm_radius*arm_radius + arm_length*arm_length) / 12}"/>

  <!-- Macros for Reusable Components -->
  <xacro:macro name="arm_link" params="prefix length radius mass">
    <link name="${prefix}_upper_arm">
      <visual>
        <geometry>
          <cylinder length="${length}" radius="${radius}"/>
        </geometry>
        <origin xyz="0 0 ${-length/2}" rpy="0 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="${length}" radius="${radius}"/>
        </geometry>
        <origin xyz="0 0 ${-length/2}" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="${mass}"/>
        <origin xyz="0 0 ${-length/2}" rpy="0 0 0"/>
        <inertia ixx="${arm_ixx}" ixy="0.0" ixz="0.0"
                 iyy="${arm_ixx}" iyz="0.0"
                 izz="${mass*radius*radius/2}"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use the Macro -->
  <xacro:arm_link prefix="left" length="${arm_length}"
                   radius="${arm_radius}" mass="${arm_mass}"/>
  <xacro:arm_link prefix="right" length="${arm_length}"
                   radius="${arm_radius}" mass="${arm_mass}"/>

</robot>
```

### Parameterized Humanoid Arm Macro

```xml
<xacro:macro name="humanoid_arm" params="prefix reflect">

  <!-- Shoulder Link -->
  <link name="${prefix}_shoulder_link">
    <visual>
      <geometry>
        <sphere radius="0.06"/>
      </geometry>
      <material name="shoulder_color">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.06"/>
      </geometry>
    </collision>
    <xacro:sphere_inertial radius="0.06" mass="0.8"/>
  </link>

  <!-- Shoulder Pitch Joint -->
  <joint name="${prefix}_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="${prefix}_shoulder_link"/>
    <origin xyz="0.0 ${reflect*0.2} 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="100" velocity="2.0"/>
    <dynamics damping="1.0" friction="0.5"/>
  </joint>

  <!-- Upper Arm Link -->
  <link name="${prefix}_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <material name="arm_color">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.04"/>
      </geometry>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    </collision>
    <xacro:cylinder_inertial length="0.3" radius="0.04" mass="1.2"/>
  </joint>

  <!-- Shoulder Roll Joint -->
  <joint name="${prefix}_shoulder_roll" type="revolute">
    <parent link="${prefix}_shoulder_link"/>
    <child link="${prefix}_upper_arm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="${-1.5*reflect}" upper="${1.5*reflect}" effort="100" velocity="2.0"/>
    <dynamics damping="1.0" friction="0.5"/>
  </joint>

  <!-- Forearm Link -->
  <link name="${prefix}_forearm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.035"/>
      </geometry>
      <origin xyz="0 0 -0.125" rpy="0 0 0"/>
      <material name="forearm_color">
        <color rgba="0.3 0.3 0.3 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.035"/>
      </geometry>
      <origin xyz="0 0 -0.125" rpy="0 0 0"/>
    </collision>
    <xacro:cylinder_inertial length="0.25" radius="0.035" mass="0.8"/>
  </link>

  <!-- Elbow Joint -->
  <joint name="${prefix}_elbow" type="revolute">
    <parent link="${prefix}_upper_arm"/>
    <child link="${prefix}_forearm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="50" velocity="2.0"/>
    <dynamics damping="0.8" friction="0.3"/>
  </joint>

  <!-- Hand Link -->
  <link name="${prefix}_hand">
    <visual>
      <geometry>
        <box size="0.08 0.12 0.02"/>
      </geometry>
      <material name="hand_color">
        <color rgba="0.9 0.8 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.08 0.12 0.02"/>
      </geometry>
    </collision>
    <xacro:box_inertial x="0.08" y="0.12" z="0.02" mass="0.3"/>
  </link>

  <!-- Wrist Joint -->
  <joint name="${prefix}_wrist" type="revolute">
    <parent link="${prefix}_forearm"/>
    <child link="${prefix}_hand"/>
    <origin xyz="0 0 -0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="2.0"/>
    <dynamics damping="0.5" friction="0.2"/>
  </joint>

</xacro:macro>

<!-- Instantiate Both Arms -->
<xacro:humanoid_arm prefix="left" reflect="1"/>
<xacro:humanoid_arm prefix="right" reflect="-1"/>
```

### Inertial Macros

```xml
<!-- Sphere Inertial -->
<xacro:macro name="sphere_inertial" params="radius mass">
  <inertial>
    <mass value="${mass}"/>
    <inertia ixx="${2*mass*radius*radius/5}" ixy="0.0" ixz="0.0"
             iyy="${2*mass*radius*radius/5}" iyz="0.0"
             izz="${2*mass*radius*radius/5}"/>
  </inertial>
</xacro:macro>

<!-- Cylinder Inertial (along Z-axis) -->
<xacro:macro name="cylinder_inertial" params="length radius mass">
  <inertial>
    <mass value="${mass}"/>
    <origin xyz="0 0 ${-length/2}" rpy="0 0 0"/>
    <inertia ixx="${mass*(3*radius*radius + length*length)/12}" ixy="0.0" ixz="0.0"
             iyy="${mass*(3*radius*radius + length*length)/12}" iyz="0.0"
             izz="${mass*radius*radius/2}"/>
  </inertial>
</xacro:macro>

<!-- Box Inertial -->
<xacro:macro name="box_inertial" params="x y z mass">
  <inertial>
    <mass value="${mass}"/>
    <inertia ixx="${mass*(y*y + z*z)/12}" ixy="0.0" ixz="0.0"
             iyy="${mass*(x*x + z*z)/12}" iyz="0.0"
             izz="${mass*(x*x + y*y)/12}"/>
  </inertial>
</xacro:macro>
```

---

## 1.4.5 Adding Sensors to URDF

### Camera Sensor

```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.02 0.05 0.02"/>
    </geometry>
    <material name="camera_color">
      <color rgba="0.1 0.1 0.1 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <box size="0.02 0.05 0.02"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.05"/>
    <inertia ixx="0.00001" ixy="0" ixz="0"
             iyy="0.00001" iyz="0" izz="0.00001"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="head_link"/>
  <child link="camera_link"/>
  <origin xyz="0.08 0 0.05" rpy="0 0 0"/>
</joint>

<!-- Gazebo Camera Plugin -->
<gazebo reference="camera_link">
  <sensor type="camera" name="head_camera">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>1920</width>
        <height>1080</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>image_raw:=camera/image_raw</remapping>
        <remapping>camera_info:=camera/camera_info</remapping>
      </ros>
      <camera_name>head_camera</camera_name>
      <frame_name>camera_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Sensor

```xml
<link name="imu_link">
  <visual>
    <geometry>
      <box size="0.03 0.03 0.01"/>
    </geometry>
    <material name="imu_color">
      <color rgba="0.2 0.2 0.8 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <box size="0.03 0.03 0.01"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.02"/>
    <inertia ixx="0.000001" ixy="0" ixz="0"
             iyy="0.000001" iyz="0" izz="0.000001"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="base_link"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.25" rpy="0 0 0"/>
</joint>

<!-- Gazebo IMU Plugin -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/out:=imu/data</remapping>
      </ros>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Sensor

```xml
<link name="lidar_link">
  <visual>
    <geometry>
      <cylinder length="0.05" radius="0.04"/>
    </geometry>
    <material name="lidar_color">
      <color rgba="0.1 0.1 0.1 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.05" radius="0.04"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.15"/>
    <inertia ixx="0.0001" ixy="0" ixz="0"
             iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="lidar_joint" type="fixed">
  <parent link="head_link"/>
  <child link="lidar_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<!-- Gazebo LiDAR Plugin -->
<gazebo reference="lidar_link">
  <sensor name="lidar_sensor" type="ray">
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/out:=lidar/scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

---

## 1.4.6 Visualization with RViz 2

### Launching URDF in RViz

**Launch File (display.launch.py):**

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Get URDF file path
    urdf_file = os.path.join(
        get_package_share_directory('my_humanoid_description'),
        'urdf',
        'humanoid.urdf.xacro'
    )

    # Process xacro file
    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )

    # Robot State Publisher Node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'publish_frequency': 30.0
        }]
    )

    # Joint State Publisher GUI (for testing)
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )

    # RViz Node
    rviz_config_file = os.path.join(
        get_package_share_directory('my_humanoid_description'),
        'rviz',
        'view_robot.rviz'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])
```

### RViz Configuration

**Minimal RViz Config (view_robot.rviz):**

```yaml
Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views

Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Plane Cell Count: 20

    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Description Topic: /robot_description
      Visual Enabled: true
      Collision Enabled: false

    - Class: rviz_default_plugins/TF
      Name: TF
      Show Names: true
      Show Axes: true
      Frame Timeout: 15

  Global Options:
    Fixed Frame: base_link

  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 2.5
      Focal Point:
        X: 0
        Y: 0
        Z: 0.5
```

### Debugging URDF

**Check URDF Syntax:**

```bash
# Validate URDF structure
check_urdf humanoid.urdf

# Convert xacro to URDF and check
xacro humanoid.urdf.xacro > humanoid_generated.urdf
check_urdf humanoid_generated.urdf
```

**Visualize TF Tree:**

```bash
# Install TF tools
sudo apt install ros-humble-tf2-tools

# View TF tree
ros2 run tf2_tools view_frames
# Generates frames.pdf showing the complete transform tree

# Echo specific transform
ros2 run tf2_ros tf2_echo base_link left_hand
```

**Common URDF Errors:**

| Error | Cause | Fix |
|-------|-------|-----|
| `No transform from X to Y` | Missing joint connection | Check parent/child links |
| `Inertia matrix not positive definite` | Invalid inertial properties | Recalculate moments of inertia |
| `Link has no inertia` | Missing `<inertial>` tag | Add mass and inertia |
| `Failed to parse URDF` | XML syntax error | Check for unclosed tags |
| `Mesh not found` | Incorrect mesh path | Verify `package://` URI |

---

## 1.4.7 Integration with robot_state_publisher

### What is robot_state_publisher?

The **robot_state_publisher** node:
- Reads the robot's URDF description
- Subscribes to `/joint_states` topic
- Publishes TF transforms for all links
- Maintains the robot's kinematic state

### Joint States Publisher

**Publishing Joint States Programmatically:**

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class HumanoidJointPublisher(Node):
    def __init__(self):
        super().__init__('humanoid_joint_publisher')

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_states)

        # Define joint names
        self.joint_names = [
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow',
            'left_hip_pitch', 'left_hip_roll', 'left_knee',
            'right_hip_pitch', 'right_hip_roll', 'right_knee'
        ]

        self.time = 0.0

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Simple sinusoidal motion for demonstration
        t = self.time
        msg.position = [
            math.sin(t),      # left_shoulder_pitch
            0.3,              # left_shoulder_roll
            math.sin(t) + 1.0,  # left_elbow
            math.sin(t),      # right_shoulder_pitch
            -0.3,             # right_shoulder_roll
            math.sin(t) + 1.0,  # right_elbow
            0.0,              # left_hip_pitch
            0.0,              # left_hip_roll
            0.5,              # left_knee
            0.0,              # right_hip_pitch
            0.0,              # right_hip_roll
            0.5               # right_knee
        ]

        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)

        self.joint_pub.publish(msg)
        self.time += 0.1

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidJointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 1.4.8 Hands-On Exercise: Create a Humanoid Leg

### Exercise Objectives

Build a complete humanoid leg URDF with:
- 6 degrees of freedom (hip, knee, ankle)
- Realistic joint limits
- Proper inertial properties
- Foot contact sensor

### Step 1: Create Package Structure

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake humanoid_leg_description
cd humanoid_leg_description
mkdir -p urdf meshes launch rviz
```

### Step 2: Create Leg URDF (urdf/leg.urdf.xacro)

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_leg">

  <!-- Properties -->
  <xacro:property name="thigh_length" value="0.4"/>
  <xacro:property name="thigh_radius" value="0.06"/>
  <xacro:property name="thigh_mass" value="5.0"/>

  <xacro:property name="shin_length" value="0.4"/>
  <xacro:property name="shin_radius" value="0.05"/>
  <xacro:property name="shin_mass" value="3.0"/>

  <xacro:property name="foot_length" value="0.25"/>
  <xacro:property name="foot_width" value="0.12"/>
  <xacro:property name="foot_height" value="0.05"/>
  <xacro:property name="foot_mass" value="1.2"/>

  <!-- Base Link (Hip) -->
  <link name="base_link">
    <visual>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
      <material name="hip_color">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0"
               iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Thigh Link -->
  <link name="thigh_link">
    <visual>
      <geometry>
        <cylinder length="${thigh_length}" radius="${thigh_radius}"/>
      </geometry>
      <origin xyz="0 0 ${-thigh_length/2}" rpy="0 0 0"/>
      <material name="thigh_color">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${thigh_length}" radius="${thigh_radius}"/>
      </geometry>
      <origin xyz="0 0 ${-thigh_length/2}" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="${thigh_mass}"/>
      <origin xyz="0 0 ${-thigh_length/2}" rpy="0 0 0"/>
      <inertia ixx="${thigh_mass*(3*thigh_radius*thigh_radius + thigh_length*thigh_length)/12}"
               ixy="0" ixz="0"
               iyy="${thigh_mass*(3*thigh_radius*thigh_radius + thigh_length*thigh_length)/12}"
               iyz="0"
               izz="${thigh_mass*thigh_radius*thigh_radius/2}"/>
    </inertial>
  </link>

  <!-- Hip Pitch Joint -->
  <joint name="hip_pitch" type="revolute">
    <parent link="base_link"/>
    <child link="thigh_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="200" velocity="2.0"/>
    <dynamics damping="2.0" friction="1.0"/>
  </joint>

  <!-- Shin Link -->
  <link name="shin_link">
    <visual>
      <geometry>
        <cylinder length="${shin_length}" radius="${shin_radius}"/>
      </geometry>
      <origin xyz="0 0 ${-shin_length/2}" rpy="0 0 0"/>
      <material name="shin_color">
        <color rgba="0.3 0.3 0.3 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${shin_length}" radius="${shin_radius}"/>
      </geometry>
      <origin xyz="0 0 ${-shin_length/2}" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="${shin_mass}"/>
      <origin xyz="0 0 ${-shin_length/2}" rpy="0 0 0"/>
      <inertia ixx="${shin_mass*(3*shin_radius*shin_radius + shin_length*shin_length)/12}"
               ixy="0" ixz="0"
               iyy="${shin_mass*(3*shin_radius*shin_radius + shin_length*shin_length)/12}"
               iyz="0"
               izz="${shin_mass*shin_radius*shin_radius/2}"/>
    </inertial>
  </link>

  <!-- Knee Joint -->
  <joint name="knee_pitch" type="revolute">
    <parent link="thigh_link"/>
    <child link="shin_link"/>
    <origin xyz="0 0 ${-thigh_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="150" velocity="2.0"/>
    <dynamics damping="1.5" friction="0.8"/>
  </joint>

  <!-- Foot Link -->
  <link name="foot_link">
    <visual>
      <geometry>
        <box size="${foot_length} ${foot_width} ${foot_height}"/>
      </geometry>
      <origin xyz="${foot_length/4} 0 ${-foot_height/2}" rpy="0 0 0"/>
      <material name="foot_color">
        <color rgba="0.2 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${foot_length} ${foot_width} ${foot_height}"/>
      </geometry>
      <origin xyz="${foot_length/4} 0 ${-foot_height/2}" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="${foot_mass}"/>
      <origin xyz="${foot_length/4} 0 ${-foot_height/2}" rpy="0 0 0"/>
      <inertia ixx="${foot_mass*(foot_width*foot_width + foot_height*foot_height)/12}"
               ixy="0" ixz="0"
               iyy="${foot_mass*(foot_length*foot_length + foot_height*foot_height)/12}"
               iyz="0"
               izz="${foot_mass*(foot_length*foot_length + foot_width*foot_width)/12}"/>
    </inertial>
  </link>

  <!-- Ankle Pitch Joint -->
  <joint name="ankle_pitch" type="revolute">
    <parent link="shin_link"/>
    <child link="foot_link"/>
    <origin xyz="0 0 ${-shin_length}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.7" upper="0.7" effort="100" velocity="2.0"/>
    <dynamics damping="1.0" friction="0.5"/>
  </joint>

  <!-- Contact Sensor (Foot Force Sensor) -->
  <gazebo reference="foot_link">
    <sensor name="foot_contact_sensor" type="contact">
      <always_on>true</always_on>
      <update_rate>100.0</update_rate>
      <contact>
        <collision>foot_link_collision</collision>
      </contact>
      <plugin name="foot_contact_plugin" filename="libgazebo_ros_bumper.so">
        <ros>
          <namespace>/humanoid</namespace>
          <remapping>bumper_states:=foot_contact</remapping>
        </ros>
        <frame_name>foot_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

### Step 3: Create Launch File (launch/display.launch.py)

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    urdf_file = os.path.join(
        get_package_share_directory('humanoid_leg_description'),
        'urdf', 'leg.urdf.xacro'
    )

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(
            get_package_share_directory('humanoid_leg_description'),
            'rviz', 'view_leg.rviz'
        )]
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz
    ])
```

### Step 4: Build and Test

```bash
cd ~/ros2_ws
colcon build --packages-select humanoid_leg_description
source install/setup.bash
ros2 launch humanoid_leg_description display.launch.py
```

**Expected Result:**
- RViz opens showing the humanoid leg model
- Joint State Publisher GUI appears with sliders for hip, knee, ankle
- Moving sliders updates the leg visualization in real-time
- TF tree shows: base_link → thigh_link → shin_link → foot_link

---

## 1.4.9 Best Practices for Humanoid URDF

### 1. Coordinate Frame Conventions

**ROS REP 103 Standard:**
- **X**: Forward (red axis)
- **Y**: Left (green axis)
- **Z**: Up (blue axis)

**Consistent Origin Placement:**
```xml
<!-- Place joint origin at the rotation point -->
<joint name="elbow" type="revolute">
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>  <!-- At end of upper arm -->
  <axis xyz="0 1 0"/>  <!-- Pitch rotation -->
</joint>
```

### 2. Realistic Inertial Properties

**Mass Distribution for Humanoids:**
- Total mass: 50-80 kg (typical adult humanoid)
- Torso: ~40% of total mass
- Each leg: ~20% of total mass
- Each arm: ~7% of total mass
- Head: ~8% of total mass

**Calculate Moments of Inertia:**
Use physics-based formulas or CAD software (SolidWorks, Blender) to export accurate inertial properties.

### 3. Joint Limits Based on Human Biomechanics

| Joint | Lower Limit | Upper Limit | Max Effort (Nm) |
|-------|-------------|-------------|-----------------|
| Shoulder Pitch | -120° (-2.09 rad) | 180° (3.14 rad) | 100 |
| Shoulder Roll | -90° (-1.57 rad) | 180° (3.14 rad) | 100 |
| Elbow | 0° (0 rad) | 150° (2.62 rad) | 50 |
| Hip Pitch | -90° (-1.57 rad) | 90° (1.57 rad) | 200 |
| Hip Roll | -45° (-0.79 rad) | 45° (0.79 rad) | 150 |
| Knee | 0° (0 rad) | 140° (2.44 rad) | 150 |
| Ankle Pitch | -45° (-0.79 rad) | 30° (0.52 rad) | 100 |

### 4. Collision Models

**Use Simplified Geometries:**
```xml
<collision>
  <geometry>
    <cylinder length="0.3" radius="0.06"/>  <!-- Simpler than visual mesh -->
  </geometry>
</collision>
```

**Self-Collision Avoidance:**
```xml
<!-- Disable collision between adjacent links -->
<gazebo>
  <static>false</static>
  <self_collide>true</self_collide>
</gazebo>

<!-- In Gazebo SDF, disable specific collisions -->
<disable_collisions link1="left_upper_arm" link2="torso"/>
```

### 5. Modular Design with Xacro

**Separate Files for Subsystems:**
```
urdf/
├── humanoid.urdf.xacro       # Main robot file
├── torso.xacro               # Torso definition
├── arm.xacro                 # Arm macro
├── leg.xacro                 # Leg macro
├── head.xacro                # Head with sensors
├── materials.xacro           # Color definitions
└── gazebo.xacro              # Gazebo-specific configs
```

**Main File Structure:**
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid">

  <xacro:include filename="$(find my_humanoid_description)/urdf/materials.xacro"/>
  <xacro:include filename="$(find my_humanoid_description)/urdf/torso.xacro"/>
  <xacro:include filename="$(find my_humanoid_description)/urdf/arm.xacro"/>
  <xacro:include filename="$(find my_humanoid_description)/urdf/leg.xacro"/>
  <xacro:include filename="$(find my_humanoid_description)/urdf/head.xacro"/>
  <xacro:include filename="$(find my_humanoid_description)/urdf/gazebo.xacro"/>

  <!-- Instantiate robot parts -->
  <xacro:torso/>
  <xacro:humanoid_arm prefix="left" reflect="1"/>
  <xacro:humanoid_arm prefix="right" reflect="-1"/>
  <xacro:humanoid_leg prefix="left" reflect="1"/>
  <xacro:humanoid_leg prefix="right" reflect="-1"/>
  <xacro:head parent="torso"/>

</robot>
```

---

## 1.4.10 Troubleshooting Common Issues

### Issue 1: Robot Falls Through Ground in Gazebo

**Cause:** Missing or incorrect collision models.

**Fix:**
```xml
<!-- Ensure all links have collision tags -->
<link name="foot_link">
  <collision>
    <geometry>
      <box size="0.25 0.12 0.05"/>
    </geometry>
  </collision>
  <!-- Add friction for ground contact -->
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
      </ode>
    </friction>
  </surface>
</link>
```

### Issue 2: Joints Not Moving

**Cause:** Joint limits too restrictive or missing joint_state_publisher.

**Fix:**
```bash
# Check if joint_states are being published
ros2 topic echo /joint_states

# Verify joint limits in URDF
<limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>

# Launch joint_state_publisher
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

### Issue 3: TF Transform Errors

**Cause:** Broken kinematic chain or duplicate frame names.

**Fix:**
```bash
# Visualize TF tree
ros2 run tf2_tools view_frames

# Check for errors
ros2 run tf2_ros tf2_echo base_link left_hand

# Ensure all joints connect properly
check_urdf humanoid.urdf
```

### Issue 4: Inertia Matrix Not Positive Definite

**Cause:** Invalid inertia tensor (non-physical values).

**Fix:**
Use automated tools or formulas:
```python
# Python script to calculate cylinder inertia
def cylinder_inertia(mass, radius, length):
    ixx = mass * (3*radius**2 + length**2) / 12
    iyy = ixx
    izz = mass * radius**2 / 2
    return ixx, iyy, izz

mass = 1.2
radius = 0.04
length = 0.3
ixx, iyy, izz = cylinder_inertia(mass, radius, length)
print(f"ixx={ixx:.6f} iyy={iyy:.6f} izz={izz:.6f}")
```

---

## Summary

In this chapter, you learned:

✅ **URDF fundamentals** for robot modeling in ROS 2
✅ **Creating humanoid URDFs** with complex kinematic chains
✅ **Using Xacro** for modular, parameterized robot descriptions
✅ **Adding sensors** (cameras, IMU, LiDAR) to URDF models
✅ **Visualizing robots** in RViz 2 and debugging TF trees
✅ **Integrating with robot_state_publisher** for real-time state tracking
✅ **Best practices** for humanoid robot description design

### Key Takeaways

1. **URDF is the foundation** for all ROS 2 robot applications
2. **Accurate inertial properties** are critical for realistic simulation
3. **Xacro macros** enable reusable, maintainable robot models
4. **Visualization tools** (RViz, TF) are essential for debugging
5. **Humanoid complexity** requires careful joint hierarchy design

---

## Next Steps

In **Chapter 2.1: Simulation Fundamentals**, you'll learn:
- Setting up Gazebo Garden for humanoid simulation
- Physics engines and contact modeling
- Sensor simulation and noise modeling
- Real-time performance optimization

**Recommended Practice:**
1. Create a full humanoid URDF with arms, legs, and torso
2. Add sensors (cameras, IMU) to your robot
3. Visualize in RViz and test joint movements
4. Prepare your URDF for Gazebo simulation (Chapter 2)

---

## Additional Resources

### Official Documentation
- [ROS 2 URDF Tutorials](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html)
- [Xacro Documentation](http://wiki.ros.org/xacro)
- [robot_state_publisher](https://github.com/ros/robot_state_publisher)

### Tools
- [check_urdf](https://github.com/ros/kdl_parser) - URDF validation
- [urdf_to_graphviz](https://github.com/ros/urdfdom) - Visualize link hierarchy
- [SolidWorks to URDF Exporter](http://wiki.ros.org/sw_urdf_exporter)

### Community Examples
- [Atlas Humanoid URDF](https://github.com/RobotLocomotion/drake/tree/master/manipulation/models/atlas)
- [Nao Humanoid URDF](https://github.com/ros-naoqi/nao_robot)
- [Pepper Robot URDF](https://github.com/ros-naoqi/pepper_robot)

---

**End of Chapter 1.4: URDF and Robot Description**
