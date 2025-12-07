---
id: chapter-03-03-robot-control-with-isaac
title: "Robot Control with Isaac"
sidebar_label: "Chapter 3: Robot Control with Isaac"
description: "Implementing advanced navigation and manipulation using Isaac Sim and Nav2"
keywords:
  - Isaac Sim
  - Navigation
  - Manipulation
  - Nav2
  - Path Planning
prerequisites:
  - chapter-01-01-architecture
  - chapter-01-02-nodes-topics-services
  - chapter-02-01-simulation-fundamentals
  - chapter-03-01-isaac-sim-fundamentals
  - chapter-03-02-isaac-ros-bridge
---


# Chapter 3 - Robot Control with Isaac

## Learning Objectives

- Implement advanced navigation systems using Isaac Sim and Nav2
- Create humanoid-specific path planning algorithms
- Configure manipulation systems with Isaac ROS
- Integrate perception and action for complex tasks
- Deploy control systems on GPU-accelerated platforms

## Prerequisites

- Completed Module 1: The Robotic Nervous System (ROS 2)
- Completed Module 2: The Digital Twin (Gazebo & Unity)
- Completed Module 3.1-3.2: Isaac Sim and Isaac ROS Bridge
- Understanding of humanoid kinematics and dynamics
- Ubuntu 22.04 LTS with ROS 2 Humble, Isaac Sim, and Isaac ROS installed
- Basic knowledge of control theory and path planning

## Introduction

Robot control with Isaac extends beyond simple navigation to encompass the full spectrum of humanoid capabilities. Isaac provides the computational framework for complex behaviors that require real-time perception, decision-making, and action execution. This chapter focuses on leveraging Isaac's capabilities for advanced navigation and manipulation in humanoid robotics applications.

Isaac's control systems integrate perception, planning, and execution in a unified framework that can handle the complexity of humanoid robots with multiple degrees of freedom and sophisticated sensor arrays.

### Why Isaac for Robot Control

Isaac enables:
- GPU-accelerated control algorithms
- Real-time perception-action loops
- Complex humanoid kinematic control
- Advanced path planning for bipedal systems
- Integration of multiple sensor modalities

### Real-world Applications

- Humanoid robots for disaster response and rescue
- Assistive robots for elderly care and support
- Industrial robots for complex assembly tasks
- Service robots for hospitality and retail

### What You'll Build by the End

By completing this chapter, you will create:
- Advanced navigation system for humanoid robots
- Manipulation control pipeline with Isaac ROS
- Integration between perception and action systems
- Control algorithms optimized for bipedal locomotion

## Core Concepts

### Humanoid Navigation Challenges

Humanoid navigation presents unique challenges:
- Bipedal locomotion dynamics
- Balance and stability requirements
- Complex obstacle avoidance for legged systems
- Multi-constraint path planning

### Isaac Control Architecture

Isaac control systems include:
- Perception-action coordination
- Real-time trajectory generation
- Dynamic balance control
- Multi-modal sensor fusion

### Manipulation Planning

Isaac enables sophisticated manipulation through:
- GPU-accelerated inverse kinematics
- Collision-aware motion planning
- Grasp planning and execution
- Tool use and dexterity

## Hands-On Tutorial

### Step 1: Create Isaac Control Package

First, create a package for Isaac control systems:

```bash
mkdir -p ~/isaac_control_ws/src/isaac_control_systems/src
cd ~/isaac_control_ws/src/isaac_control_systems
```

Create `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>isaac_control_systems</name>
  <version>0.0.0</version>
  <description>Isaac control systems for humanoid robotics</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>moveit_msgs</depend>
  <depend>control_msgs</depend>
  <depend>trajectory_msgs</depend>
  <depend>isaac_ros_visual_slam</depend>
  <depend>isaac_ros_apriltag</depend>
  <depend>nav2_msgs</depend>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(isaac_control_systems)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(isaac_ros_visual_slam REQUIRED)
find_package(isaac_ros_apriltag REQUIRED)
find_package(nav2_msgs REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  isaac_control_systems/humanoid_controller.py
  isaac_control_systems/manipulation_planner.py
  isaac_control_systems/navigation_manager.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
```

### Step 2: Create Humanoid Controller Node

Create `isaac_control_systems/humanoid_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from tf2_ros import TransformListener, Buffer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers
        self.joint_trajectory_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.balance_status_pub = self.create_publisher(Bool, 'balance_status', 10)
        self.control_metrics_pub = self.create_publisher(Float32, 'control_metrics', 10)
        self.status_pub = self.create_publisher(String, 'controller_status', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.desired_pose_sub = self.create_subscription(
            Pose, '/move_base_simple/goal', self.desired_pose_callback, 10)

        # Initialize TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Humanoid state
        self.joint_positions = {}
        self.joint_velocities = {}
        self.imu_data = None
        self.odom_data = None
        self.desired_pose = None
        self.current_balance = 0.0
        self.balance_threshold = 0.8

        # Control parameters
        self.control_frequency = 100  # Hz
        self.max_joint_velocity = 2.0  # rad/s
        self.balance_kp = 1.0
        self.balance_kd = 0.1

        # Humanoid kinematic model (simplified)
        self.humanoid_model = {
            'joints': [
                'left_hip', 'left_knee', 'left_ankle',
                'right_hip', 'right_knee', 'right_ankle',
                'left_shoulder', 'left_elbow', 'left_wrist',
                'right_shoulder', 'right_elbow', 'right_wrist'
            ],
            'limits': {
                'hip': {'min': -1.57, 'max': 1.57},
                'knee': {'min': 0, 'max': 2.5},
                'ankle': {'min': -0.5, 'max': 0.5},
                'shoulder': {'min': -2.0, 'max': 2.0},
                'elbow': {'min': -2.0, 'max': 0.5},
                'wrist': {'min': -1.57, 'max': 1.57}
            }
        }

        # Balance control state
        self.balance_error_history = []
        self.max_history = 10

        # Timer for control loop
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_loop)

        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        """Process joint state data"""
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            self.joint_positions[name] = pos
            self.joint_velocities[name] = vel

    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        self.imu_data = msg
        # Calculate current balance based on IMU data
        self.current_balance = self.calculate_balance_from_imu()

    def odom_callback(self, msg):
        """Process odometry data"""
        self.odom_data = msg

    def desired_pose_callback(self, msg):
        """Process desired pose for navigation"""
        self.desired_pose = msg

    def calculate_balance_from_imu(self):
        """Calculate balance metric from IMU data"""
        if self.imu_data is None:
            return 0.0

        # Calculate balance based on orientation and angular velocity
        quat = self.imu_data.orientation
        ang_vel = self.imu_data.angular_velocity

        # Convert quaternion to roll/pitch angles
        rotation = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        euler = rotation.as_euler('xyz')

        # Balance metric: 1.0 is perfectly balanced, 0.0 is fallen
        roll_balance = max(0.0, min(1.0, 1.0 - abs(euler[0]) / (np.pi/3)))  # Max 60 deg roll
        pitch_balance = max(0.0, min(1.0, 1.0 - abs(euler[1]) / (np.pi/3)))  # Max 60 deg pitch
        angular_balance = max(0.0, min(1.0, 1.0 - np.sqrt(ang_vel.x**2 + ang_vel.y**2) / 2.0))  # Max 2 rad/s

        balance = (roll_balance + pitch_balance + angular_balance) / 3.0
        return balance

    def calculate_balance_control(self):
        """Calculate balance control adjustments"""
        if self.imu_data is None:
            return {}

        # Calculate balance error
        balance_error = self.balance_threshold - self.current_balance
        self.balance_error_history.append(balance_error)
        if len(self.balance_error_history) > self.max_history:
            self.balance_error_history.pop(0)

        # Calculate derivative of error
        if len(self.balance_error_history) >= 2:
            balance_error_deriv = (self.balance_error_history[-1] - self.balance_error_history[-2]) * self.control_frequency
        else:
            balance_error_deriv = 0.0

        # PID control for balance
        balance_control = (self.balance_kp * balance_error +
                          self.balance_kd * balance_error_deriv)

        # Calculate joint adjustments based on balance control
        joint_adjustments = {}

        # Adjust leg joints for balance
        for leg in ['left', 'right']:
            hip_joint = f'{leg}_hip'
            knee_joint = f'{leg}_knee'
            ankle_joint = f'{leg}_ankle'

            if hip_joint in self.joint_positions:
                # Adjust hip based on balance needs
                hip_adjustment = balance_control * (1.0 if leg == 'left' else -1.0) * 0.1
                joint_adjustments[hip_joint] = self.joint_positions[hip_joint] + hip_adjustment

            if knee_joint in self.joint_positions:
                # Adjust knee for balance
                knee_adjustment = balance_control * 0.05
                joint_adjustments[knee_joint] = self.joint_positions[knee_joint] + knee_adjustment

            if ankle_joint in self.joint_positions:
                # Adjust ankle for balance
                ankle_adjustment = balance_control * (1.0 if leg == 'left' else -1.0) * 0.05
                joint_adjustments[ankle_joint] = self.joint_positions[ankle_joint] + ankle_adjustment

        return joint_adjustments

    def generate_locomotion_trajectory(self):
        """Generate walking trajectory for humanoid"""
        if self.desired_pose is None or self.odom_data is None:
            return None

        # Calculate desired movement direction
        current_pos = self.odom_data.pose.pose.position
        desired_pos = self.desired_pose.position

        dx = desired_pos.x - current_pos.x
        dy = desired_pos.y - current_pos.y
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.1:  # Close enough to target
            return None

        # Simple walking pattern generation (simplified)
        trajectory = JointTrajectory()
        trajectory.joint_names = self.humanoid_model['joints']

        # Generate trajectory points for walking
        num_points = 20
        for i in range(num_points):
            point = JointTrajectoryPoint()

            # Time from start
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int((i + 1) * (0.5 / num_points) * 1e9)  # 0.5 second steps

            # Calculate joint positions for walking gait
            phase = (i / num_points) * 2 * np.pi

            positions = []
            for joint_name in self.humanoid_model['joints']:
                if 'hip' in joint_name:
                    # Hip oscillation for walking
                    base_pos = self.joint_positions.get(joint_name, 0.0)
                    oscillation = 0.2 * np.sin(phase + (0.5 if 'right' in joint_name else 0))
                    positions.append(base_pos + oscillation)
                elif 'knee' in joint_name:
                    # Knee bending for walking
                    base_pos = self.joint_positions.get(joint_name, 0.0)
                    bend = 0.3 * np.sin(phase + (0.5 if 'right' in joint_name else 0))**2
                    positions.append(base_pos + bend)
                elif 'ankle' in joint_name:
                    # Ankle adjustment for balance
                    base_pos = self.joint_positions.get(joint_name, 0.0)
                    adjustment = 0.1 * np.sin(phase + (0.5 if 'right' in joint_name else 0))
                    positions.append(base_pos + adjustment)
                elif 'shoulder' in joint_name:
                    # Arm swing for balance
                    base_pos = self.joint_positions.get(joint_name, 0.0)
                    swing = 0.15 * np.sin(phase + (0.5 if 'right' in joint_name else 0) + np.pi)
                    positions.append(base_pos + swing)
                elif 'elbow' in joint_name:
                    # Elbow bend during walking
                    base_pos = self.joint_positions.get(joint_name, 0.0)
                    bend = 0.1 * np.sin(phase)
                    positions.append(base_pos + bend)
                else:
                    # Other joints maintain current position
                    positions.append(self.joint_positions.get(joint_name, 0.0))

            point.positions = positions
            # Set velocities to encourage smooth motion
            point.velocities = [0.0] * len(positions)

            trajectory.points.append(point)

        return trajectory

    def control_loop(self):
        """Main control loop for humanoid robot"""
        start_time = time.time()

        # Calculate balance control adjustments
        balance_adjustments = self.calculate_balance_control()

        # Generate locomotion trajectory if target is set
        locomotion_trajectory = self.generate_locomotion_trajectory()

        # Combine balance adjustments with locomotion
        if locomotion_trajectory is not None:
            # Apply balance adjustments to trajectory
            for point in locomotion_trajectory.points:
                for i, joint_name in enumerate(locomotion_trajectory.joint_names):
                    if joint_name in balance_adjustments:
                        point.positions[i] += balance_adjustments[joint_name]

        # Publish trajectory if available
        if locomotion_trajectory is not None:
            self.joint_trajectory_pub.publish(locomotion_trajectory)

        # Publish balance status
        balance_status = Bool()
        balance_status.data = self.current_balance > self.balance_threshold
        self.balance_status_pub.publish(balance_status)

        # Publish control metrics
        metrics_msg = Float32()
        metrics_msg.data = self.current_balance
        self.control_metrics_pub.publish(metrics_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f"Balance: {self.current_balance:.3f}, Stable: {balance_status.data}"
        self.status_pub.publish(status_msg)

        # Calculate control loop performance
        loop_time = time.time() - start_time
        self.get_logger().debug(f'Control loop time: {loop_time:.4f}s')

    def get_balance_metrics(self):
        """Get current balance and stability metrics"""
        return {
            'balance_score': self.current_balance,
            'balance_threshold': self.balance_threshold,
            'is_stable': self.current_balance > self.balance_threshold,
            'control_frequency': self.control_frequency
        }

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Humanoid controller stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 3: Create Manipulation Planner

Create `isaac_control_systems/manipulation_planner.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Vector3
from std_msgs.msg import Float32, Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import MoveItErrorCodes, PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from tf2_ros import TransformListener, Buffer
from cv_bridge import CvBridge
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

class ManipulationPlanner(Node):
    def __init__(self):
        super().__init__('manipulation_planner')

        # Publishers
        self.arm_trajectory_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_command_pub = self.create_publisher(Float32, '/gripper/command', 10)
        self.planning_scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self.manipulation_status_pub = self.create_publisher(String, 'manipulation_status', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.camera_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, 10)

        # Initialize components
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Manipulation state
        self.joint_positions = {}
        self.camera_image = None
        self.camera_info = None
        self.manipulation_targets = []
        self.current_task = None
        self.manipulation_active = False

        # Manipulation parameters
        self.reach_threshold = 0.1  # 10cm reach threshold
        self.gripper_open_pos = 0.05  # Fully open
        self.gripper_close_pos = 0.01  # Object grasp position

        # Object detection state
        self.detected_objects = []
        self.object_poses = {}

        # Timer for manipulation planning
        self.manipulation_timer = self.create_timer(0.1, self.manipulation_planning_loop)

        self.get_logger().info('Manipulation Planner initialized')

    def joint_state_callback(self, msg):
        """Process joint state data"""
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def camera_callback(self, msg):
        """Process camera data for object detection"""
        try:
            self.camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Camera callback error: {e}')

    def camera_info_callback(self, msg):
        """Process camera calibration data"""
        self.camera_info = msg

    def detect_objects_in_image(self):
        """Detect objects in camera image (simplified approach)"""
        if self.camera_image is None:
            return []

        # In a real implementation, this would use Isaac ROS perception
        # For this example, we'll simulate object detection
        detected_objects = []

        # Simulate detection of a few objects
        # In reality, this would use Isaac ROS object detection
        simulated_objects = [
            {'name': 'object1', 'x': 0.5, 'y': 0.2, 'z': 0.8, 'type': 'box'},
            {'name': 'object2', 'x': 0.7, 'y': -0.1, 'z': 0.9, 'type': 'cylinder'},
        ]

        return simulated_objects

    def transform_object_to_robot_frame(self, object_pose):
        """Transform object pose from camera frame to robot base frame"""
        try:
            # In a real implementation, this would use TF transforms
            # For simulation, we'll assume a fixed transform
            transformed_pose = Pose()
            transformed_pose.position.x = object_pose['x'] + 0.1  # Adjust for camera offset
            transformed_pose.position.y = object_pose['y']
            transformed_pose.position.z = object_pose['z'] - 0.2
            transformed_pose.orientation.w = 1.0
            return transformed_pose
        except Exception as e:
            self.get_logger().error(f'Transform error: {e}')
            return None

    def plan_arm_trajectory_to_object(self, object_pose):
        """Plan arm trajectory to reach an object"""
        if not self.joint_positions:
            return None

        # Define arm joint names (example for a simple 6-DOF arm)
        arm_joints = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # Check if all required joints are present
        for joint in arm_joints:
            if joint not in self.joint_positions:
                self.get_logger().warn(f'Missing joint: {joint}')
                return None

        # Create trajectory to object
        trajectory = JointTrajectory()
        trajectory.joint_names = arm_joints

        # Calculate inverse kinematics solution (simplified)
        # In a real implementation, this would use MoveIt or Isaac's IK solver
        target_positions = self.calculate_ik_solution(object_pose, arm_joints)

        if target_positions is not None:
            # Create trajectory point
            point = JointTrajectoryPoint()
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(2e9)  # 2 seconds

            # Set positions - interpolate from current to target
            current_positions = [self.joint_positions[joint] for joint in arm_joints]

            # Simple linear interpolation
            for i in range(len(arm_joints)):
                point.positions.append(current_positions[i] +
                                     0.5 * (target_positions[i] - current_positions[i]))
                point.velocities.append(0.0)  # Start with zero velocity
                point.accelerations.append(0.0)  # Start with zero acceleration

            trajectory.points.append(point)

            return trajectory

        return None

    def calculate_ik_solution(self, target_pose, joint_names):
        """Calculate inverse kinematics solution (simplified)"""
        # This is a very simplified IK solver for demonstration
        # In practice, use MoveIt or Isaac's IK solvers

        # Extract target position
        target_x = target_pose.position.x
        target_y = target_pose.position.y
        target_z = target_pose.position.z

        # Simple geometric IK for a 3-DOF arm (shoulder, elbow, wrist)
        # This is highly simplified and would need proper IK solver in practice
        try:
            # Calculate joint angles to reach target (approximate)
            joint_angles = [0.0] * len(joint_names)

            # Shoulder pan: angle to target in XY plane
            joint_angles[0] = np.arctan2(target_y, target_x)

            # Shoulder lift and elbow: approximate for reaching target
            dist_xy = np.sqrt(target_x**2 + target_y**2)
            dist_3d = np.sqrt(dist_xy**2 + target_z**2)

            # Simple approximation - in reality need proper 3D IK
            joint_angles[1] = np.arctan2(target_z, dist_xy)  # Shoulder lift
            joint_angles[2] = 0.0  # Elbow - would need proper calculation

            # Set remaining joints to default positions
            for i in range(3, len(joint_angles)):
                joint_angles[i] = 0.0

            return joint_angles

        except Exception as e:
            self.get_logger().error(f'IK calculation error: {e}')
            return None

    def create_grasp_trajectory(self, pre_grasp_pose, grasp_pose):
        """Create trajectory for grasping an object"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ['gripper_finger_joint']  # Example gripper joint

        # Pre-grasp position (open gripper)
        pre_grasp_point = JointTrajectoryPoint()
        pre_grasp_point.time_from_start.sec = 0
        pre_grasp_point.time_from_start.nanosec = int(1e9)  # 1 second
        pre_grasp_point.positions = [self.gripper_open_pos]
        pre_grasp_point.velocities = [0.0]

        # Grasp position (close gripper)
        grasp_point = JointTrajectoryPoint()
        grasp_point.time_from_start.sec = 0
        grasp_point.time_from_start.nanosec = int(2e9)  # 2 seconds
        grasp_point.positions = [self.gripper_close_pos]
        grasp_point.velocities = [0.0]

        trajectory.points = [pre_grasp_point, grasp_point]
        return trajectory

    def manipulation_planning_loop(self):
        """Main manipulation planning loop"""
        # Detect objects in current view
        detected_objects = self.detect_objects_in_image()

        if detected_objects:
            # Transform object poses to robot frame
            for obj in detected_objects:
                obj_pose = self.transform_object_to_robot_frame(obj)
                if obj_pose:
                    self.object_poses[obj['name']] = obj_pose
                    self.get_logger().info(f'Detected object: {obj["name"]} at {obj_pose.position.x:.2f}, {obj_pose.position.y:.2f}, {obj_pose.position.z:.2f}')

        # Check if we have a manipulation task
        if self.manipulation_active and self.current_task:
            task = self.current_task

            if task['type'] == 'grasp':
                # Plan trajectory to object
                if task['object'] in self.object_poses:
                    obj_pose = self.object_poses[task['object']]

                    # Check if object is reachable
                    current_effector_pos = self.get_current_end_effector_position()
                    distance = np.sqrt(
                        (obj_pose.position.x - current_effector_pos.x)**2 +
                        (obj_pose.position.y - current_effector_pos.y)**2 +
                        (obj_pose.position.z - current_effector_pos.z)**2
                    )

                    if distance < self.reach_threshold:
                        # Object is reachable, execute grasp
                        grasp_traj = self.create_grasp_trajectory(None, obj_pose)
                        if grasp_traj:
                            self.arm_trajectory_pub.publish(grasp_traj)
                            self.get_logger().info(f'Executing grasp for {task["object"]}')
                    else:
                        # Plan trajectory to reach object
                        reach_traj = self.plan_arm_trajectory_to_object(obj_pose)
                        if reach_traj:
                            self.arm_trajectory_pub.publish(reach_traj)
                            self.get_logger().info(f'Planning reach trajectory for {task["object"]}')

        # Publish manipulation status
        status_msg = String()
        status_msg.data = f"Objects detected: {len(self.object_poses)}, Active: {self.manipulation_active}"
        self.manipulation_status_pub.publish(status_msg)

    def get_current_end_effector_position(self):
        """Get current end effector position (simplified)"""
        # In a real implementation, this would calculate FK or get from TF
        # For simulation, return a placeholder
        pos = Point()
        pos.x = 0.5  # Placeholder position
        pos.y = 0.0
        pos.z = 0.8
        return pos

    def start_manipulation_task(self, task_type, target_object):
        """Start a manipulation task"""
        self.current_task = {
            'type': task_type,
            'object': target_object,
            'start_time': time.time()
        }
        self.manipulation_active = True
        self.get_logger().info(f'Started manipulation task: {task_type} {target_object}')

    def stop_manipulation(self):
        """Stop current manipulation task"""
        self.manipulation_active = False
        self.current_task = None
        self.get_logger().info('Stopped manipulation')

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Manipulation planner stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Create Navigation Manager

Create `isaac_control_systems/navigation_manager.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, Imu
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from std_msgs.msg import Float32, Bool, String
from tf2_ros import TransformListener, Buffer
from visualization_msgs.msg import MarkerArray
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

class NavigationManager(Node):
    def __init__(self):
        super().__init__('navigation_manager')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.global_plan_pub = self.create_publisher(Path, '/plan', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal', 10)
        self.navigation_status_pub = self.create_publisher(String, 'navigation_status', 10)
        self.velocity_limit_pub = self.create_publisher(Float32, 'velocity_limit', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Initialize TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation state
        self.current_pose = None
        self.current_velocity = None
        self.scan_data = None
        self.occupancy_map = None
        self.imu_data = None
        self.navigation_goal = None
        self.global_path = []
        self.local_path = []
        self.path_index = 0

        # Navigation parameters
        self.linear_vel_max = 0.5  # m/s
        self.angular_vel_max = 0.5  # rad/s
        self.min_distance_to_goal = 0.2  # m
        self.path_lookahead = 1.0  # m
        self.obstacle_threshold = 0.5  # m
        self.inflation_radius = 0.3  # m

        # Safety and control state
        self.emergency_stop = False
        self.navigation_active = False
        self.path_following = False
        self.avoiding_obstacles = False

        # Timing
        self.control_frequency = 10  # Hz
        self.last_command_time = time.time()

        # Timer for navigation control
        self.nav_timer = self.create_timer(1.0/self.control_frequency, self.navigation_control_loop)

        self.get_logger().info('Navigation Manager initialized')

    def odom_callback(self, msg):
        """Process odometry data"""
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist

    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        self.scan_data = msg
        # Check for immediate obstacles
        if self.current_velocity and self.current_velocity.linear.x > 0.1:
            # Check forward direction (simplified)
            front_scan = msg.ranges[len(msg.ranges)//2-10:len(msg.ranges)//2+10]
            min_front_dist = min([r for r in front_scan if r > msg.range_min and r < msg.range_max])
            if min_front_dist < self.obstacle_threshold:
                self.emergency_stop = True
                self.get_logger().warn(f'Obstacle detected: {min_front_dist:.2f}m ahead')
            else:
                self.emergency_stop = False

    def map_callback(self, msg):
        """Process occupancy grid map"""
        self.occupancy_map = msg

    def imu_callback(self, msg):
        """Process IMU data for navigation safety"""
        self.imu_data = msg

    def set_navigation_goal(self, x, y, theta=0.0):
        """Set navigation goal"""
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = np.sin(theta / 2.0)
        goal.pose.orientation.w = np.cos(theta / 2.0)

        self.navigation_goal = goal
        self.navigation_active = True
        self.path_following = False
        self.global_path = []
        self.local_path = []
        self.path_index = 0

        # Plan global path
        self.plan_global_path()

        self.get_logger().info(f'Navigation goal set: ({x:.2f}, {y:.2f})')

    def plan_global_path(self):
        """Plan global path using A* or Dijkstra (simplified)"""
        if not self.current_pose or not self.navigation_goal:
            return

        # Simplified path planning - in reality would use Nav2 or custom planner
        start = np.array([self.current_pose.position.x, self.current_pose.position.y])
        goal = np.array([self.navigation_goal.pose.position.x, self.navigation_goal.pose.position.y])

        # Create straight-line path as a simple example
        path = []
        steps = max(10, int(np.linalg.norm(goal - start) / 0.1))  # 10cm resolution

        for i in range(steps + 1):
            t = i / steps
            point = start + t * (goal - start)

            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            path.append(pose)

        self.global_path = path
        self.path_following = True

        # Publish global plan
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        path_msg.poses = path
        self.global_plan_pub.publish(path_msg)

    def plan_local_path(self):
        """Plan local path considering obstacles"""
        if not self.global_path or not self.current_pose:
            return []

        # Find path segment near robot
        robot_pos = np.array([self.current_pose.position.x, self.current_pose.position.y])

        # Find closest point on global path
        closest_idx = 0
        min_dist = float('inf')
        for i, pose in enumerate(self.global_path):
            path_pos = np.array([pose.pose.position.x, pose.pose.position.y])
            dist = np.linalg.norm(robot_pos - path_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Create local path from current position to look-ahead point
        local_path = []
        start_idx = closest_idx
        look_ahead = self.path_lookahead

        current_pos = robot_pos
        for i in range(start_idx, min(len(self.global_path), start_idx + 20)):
            path_pos = np.array([self.global_path[i].pose.position.x,
                                self.global_path[i].pose.position.y])

            # Check if this point is within look-ahead distance
            if np.linalg.norm(current_pos - path_pos) > look_ahead:
                break

            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = path_pos[0]
            pose.pose.position.y = path_pos[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            local_path.append(pose)
            current_pos = path_pos

        return local_path

    def calculate_navigation_command(self):
        """Calculate navigation command based on current state"""
        if not self.path_following or not self.global_path or not self.current_pose:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd

        # Get local path
        self.local_path = self.plan_local_path()

        if not self.local_path:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd

        # Get next point on path
        next_point = self.local_path[min(1, len(self.local_path)-1)]
        robot_pos = np.array([self.current_pose.position.x, self.current_pose.position.y])
        next_pos = np.array([next_point.pose.position.x, next_point.pose.position.y])

        # Calculate direction to next point
        direction = next_pos - robot_pos
        distance_to_next = np.linalg.norm(direction)

        cmd = Twist()

        if distance_to_next > 0.05:  # If not very close to next point
            # Normalize direction
            direction_norm = direction / distance_to_next

            # Calculate desired orientation
            desired_yaw = np.arctan2(direction_norm[1], direction_norm[0])

            # Get current orientation
            current_quat = self.current_pose.orientation
            current_rot = R.from_quat([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
            current_euler = current_rot.as_euler('xyz')
            current_yaw = current_euler[2]

            # Calculate orientation error
            yaw_error = desired_yaw - current_yaw
            # Normalize angle to [-pi, pi]
            while yaw_error > np.pi:
                yaw_error -= 2 * np.pi
            while yaw_error < -np.pi:
                yaw_error += 2 * np.pi

            # Set velocities
            cmd.linear.x = min(self.linear_vel_max, max(0.0, distance_to_next * 0.5))  # Proportional to distance
            cmd.angular.z = min(self.angular_vel_max, max(-self.angular_vel_max, yaw_error * 1.0))  # Proportional to error
        else:
            # Very close to path point, just orient to next segment
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Check for obstacles and adjust if needed
        if self.scan_data and self.current_velocity:
            front_scan = self.scan_data.ranges[len(self.scan_data.ranges)//2-10:len(self.scan_data.ranges)//2+10]
            min_front_dist = min([r for r in front_scan if r > self.scan_data.range_min and r < self.scan_data.range_max])

            if min_front_dist < self.obstacle_threshold:
                # Slow down or stop for obstacles
                cmd.linear.x = max(0.0, cmd.linear.x * (min_front_dist / self.obstacle_threshold))
                self.avoiding_obstacles = True
            else:
                self.avoiding_obstacles = False

        return cmd

    def navigation_control_loop(self):
        """Main navigation control loop"""
        if self.emergency_stop:
            # Emergency stop - publish zero velocity
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            self.get_logger().warn('Emergency stop activated')
            return

        if not self.navigation_active:
            return

        # Calculate navigation command
        cmd = self.calculate_navigation_command()

        # Check if goal is reached
        if self.navigation_goal and self.current_pose:
            goal_pos = np.array([self.navigation_goal.pose.position.x, self.navigation_goal.pose.position.y])
            robot_pos = np.array([self.current_pose.position.x, self.current_pose.position.y])
            distance_to_goal = np.linalg.norm(goal_pos - robot_pos)

            if distance_to_goal < self.min_distance_to_goal:
                # Goal reached
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.navigation_active = False
                self.path_following = False
                self.get_logger().info(f'Goal reached! Distance: {distance_to_goal:.3f}m')

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Publish velocity limit for safety
        vel_limit = Float32()
        vel_limit.data = self.linear_vel_max if not self.avoiding_obstacles else self.linear_vel_max * 0.3
        self.velocity_limit_pub.publish(vel_limit)

        # Publish navigation status
        status_msg = String()
        if self.navigation_active:
            status = f"Nav: Vel=({cmd.linear.x:.2f}, {cmd.angular.z:.2f}), DistToGoal={distance_to_goal:.2f}m, Obstacles={self.avoiding_obstacles}"
        else:
            status = "Navigation idle"
        status_msg.data = status
        self.navigation_status_pub.publish(status_msg)

    def get_navigation_metrics(self):
        """Get current navigation performance metrics"""
        return {
            'navigation_active': self.navigation_active,
            'path_following': self.path_following,
            'obstacle_avoidance_active': self.avoiding_obstacles,
            'control_frequency': self.control_frequency,
            'emergency_stop': self.emergency_stop
        }

def main(args=None):
    rclpy.init(args=args)
    node = NavigationManager()

    try:
        # Example: Set a navigation goal after initialization
        time.sleep(1)  # Allow subscribers to connect
        node.set_navigation_goal(2.0, 2.0)  # Navigate to (2m, 2m)

        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Navigation manager stopped by user')
    finally:
        # Stop robot before shutting down
        stop_cmd = Twist()
        node.cmd_vel_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

