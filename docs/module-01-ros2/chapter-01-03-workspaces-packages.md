---
id: chapter-01-03-workspaces-packages
title: Workspaces and Packages
sidebar_label: Workspaces and Packages
description: Creating and managing ROS 2 workspaces and packages for project organization
keywords:
  - ROS 2
  - Workspaces
  - Packages
  - Project Structure
  - Build System
prerequisites:
  - chapter-01-01-architecture
  - chapter-01-02-nodes-topics-services
---


# Building ROS 2 Packages and Workspaces

## Learning Objectives

- Create and manage ROS 2 workspaces for project organization
- Build custom ROS 2 packages with proper structure
- Implement nodes, messages, services, and actions in packages
- Use colcon build system effectively
- Debug common build and packaging issues

## Prerequisites

- Understanding of ROS 2 nodes, topics, and services
- Completed Chapter 2: Nodes, Topics, and Services in Depth
- Ubuntu 22.04 LTS with ROS 2 Humble installed
- Basic knowledge of C++ and Python
- Familiarity with version control systems

## Introduction

ROS 2 packages and workspaces provide the organizational structure for robotic software development. A workspace is a directory that contains multiple packages, while a package is the basic building unit that contains source code, dependencies, and configuration files.

This chapter will guide you through creating properly structured ROS 2 packages and managing workspaces for efficient development workflows. Understanding these concepts is essential for building maintainable and reusable robotic applications.

### Why Proper Package Structure Matters

Well-structured packages ensure code reusability, simplify dependency management, and enable collaborative development. The standard package structure also makes it easier for other developers to understand and contribute to your projects.

### Real-world Applications

- Robot application frameworks that combine multiple packages
- Reusable sensor drivers and control libraries
- Simulation environments with modular components
- Multi-robot systems with shared packages

### What You'll Build by the End

By completing this chapter, you will create:
- A complete ROS 2 workspace with multiple packages
- A custom message package with custom message definitions
- A publisher/subscriber package demonstrating communication
- A service package with custom service definitions
- A launch file to coordinate the entire system

## Core Concepts

### Workspace Structure

A ROS 2 workspace typically follows the structure: `workspace_root/src/` where source packages are placed, and build artifacts are stored in `build/`, `install/`, and `log/` directories.

### Package Structure

A standard ROS 2 package contains:
- `package.xml`: Package metadata and dependencies
- `CMakeLists.txt`: Build configuration for C++ packages
- `setup.py`: Python package configuration
- `src/`: Source code files
- `include/`: Header files (C++)
- `msg/`: Custom message definitions
- `srv/`: Custom service definitions
- `action/`: Custom action definitions
- `launch/`: Launch files for system orchestration

### Build System (Colcon)

Colcon is the build tool for ROS 2 that can handle multiple package build systems simultaneously. It's designed to be efficient and flexible for building complex robotic software systems.

## Hands-On Tutorial

### Step 1: Create a New ROS 2 Workspace

First, create a new workspace specifically for this chapter:

```bash
# Create a new workspace directory
mkdir -p ~/ros2_workspace_demo/src
cd ~/ros2_workspace_demo

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Build the empty workspace to verify setup
colcon build

# Source the workspace
source install/setup.bash

# Verify workspace is set up correctly
echo $AMENT_PREFIX_PATH
```

Expected output: The workspace path should be included in the AMENT_PREFIX_PATH environment variable.

### Step 2: Create a Custom Message Package

Create a package for custom messages that our other packages will use:

```bash
cd ~/ros2_workspace_demo/src

# Create a message package
ros2 pkg create --name robot_msgs --dependencies std_msgs geometry_msgs builtin_interfaces

# Navigate to the package directory
cd robot_msgs
```

Now edit the `package.xml` to add message generation dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robot_msgs</name>
  <version>0.0.0</version>
  <description>Custom messages for robot communication</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <build_depend>builtin_interfaces</build_depend>
  <build_depend>std_msgs</build_depend>
  <build_depend>geometry_msgs</build_depend>
  <build_depend>rosidl_default_generators</build_depend>

  <exec_depend>builtin_interfaces</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>

  <member_of_group>rosidl_interface_packages</member_of_group>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create the `msg` directory and add a custom message:

```bash
mkdir msg
```

Create `msg/RobotStatus.msg`:

```bash
# Robot status message
bool active
float32 battery_level
string current_task
int32 error_code
```

Create `msg/RobotCommand.msg`:

```bash
# Robot command message
float32 linear_velocity
float32 angular_velocity
bool emergency_stop
string task_name
```

Create `srv/TaskService.srv`:

```bash
# Request
string task_name
int32 priority
---
# Response
bool success
string message
```

Update the `CMakeLists.txt` to include message generation:

```cmake
cmake_minimum_required(VERSION 3.8)
project(robot_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages
set(msg_files
  "msg/RobotStatus.msg"
  "msg/RobotCommand.msg"
)

set(srv_files
  "srv/TaskService.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

ament_package()
```

### Step 3: Create a Publisher Package

Now create a package that uses the custom messages:

```bash
cd ~/ros2_workspace_demo/src

# Create a robot controller package
ros2 pkg create --name robot_controller --dependencies rclcpp rclpy std_msgs robot_msgs geometry_msgs

# Navigate to the package
cd robot_controller
```

Edit the `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robot_controller</name>
  <version>0.0.0</version>
  <description>Robot controller package</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>robot_msgs</depend>
  <depend>geometry_msgs</depend>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>ament_index_python</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create the source files directory:

```bash
mkdir -p src
```

Create `src/robot_publisher.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from robot_msgs.msg import RobotStatus, RobotCommand
from geometry_msgs.msg import Twist
import random

class RobotPublisher(Node):
    def __init__(self):
        super().__init__('robot_publisher')

        # Publishers
        self.status_pub = self.create_publisher(RobotStatus, 'robot_status', 10)
        self.cmd_pub = self.create_publisher(RobotCommand, 'robot_commands', 10)
        self.velocity_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Timer to publish data
        self.timer = self.create_timer(0.5, self.publish_data)
        self.status_counter = 0

        self.get_logger().info('Robot Publisher initialized')

    def publish_data(self):
        # Publish robot status
        status_msg = RobotStatus()
        status_msg.active = True
        status_msg.battery_level = random.uniform(20.0, 100.0)
        status_msg.current_task = f'Task_{self.status_counter % 5}'
        status_msg.error_code = 0
        self.status_pub.publish(status_msg)

        # Publish robot command
        cmd_msg = RobotCommand()
        cmd_msg.linear_velocity = random.uniform(0.1, 0.5)
        cmd_msg.angular_velocity = random.uniform(-0.3, 0.3)
        cmd_msg.emergency_stop = False
        cmd_msg.task_name = f'Command_{self.status_counter}'
        self.cmd_pub.publish(cmd_msg)

        # Publish velocity command
        vel_msg = Twist()
        vel_msg.linear.x = cmd_msg.linear_velocity
        vel_msg.angular.z = cmd_msg.angular_velocity
        self.velocity_pub.publish(vel_msg)

        self.status_counter += 1
        self.get_logger().info(f'Published status: {status_msg.current_task}, battery: {status_msg.battery_level:.1f}%')

def main(args=None):
    rclpy.init(args=args)
    node = RobotPublisher()

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

Create `src/robot_subscriber.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from robot_msgs.msg import RobotStatus, RobotCommand
from geometry_msgs.msg import Twist
import random

class RobotSubscriber(Node):
    def __init__(self):
        super().__init__('robot_subscriber')

        # Subscribers
        self.status_sub = self.create_subscription(
            RobotStatus, 'robot_status', self.status_callback, 10)
        self.cmd_sub = self.create_subscription(
            RobotCommand, 'robot_commands', self.cmd_callback, 10)
        self.vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.velocity_callback, 10)

        self.get_logger().info('Robot Subscriber initialized')

    def status_callback(self, msg):
        self.get_logger().info(f'Robot Status - Active: {msg.active}, Battery: {msg.battery_level:.1f}%, Task: {msg.current_task}')

    def cmd_callback(self, msg):
        self.get_logger().info(f'Robot Command - Linear: {msg.linear_velocity:.2f}, Angular: {msg.angular_velocity:.2f}, Task: {msg.task_name}')

    def velocity_callback(self, msg):
        self.get_logger().info(f'Velocity Command - Linear: {msg.linear.x:.2f}, Angular: {msg.angular.z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = RobotSubscriber()

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

Update the `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(robot_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(robot_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  src/robot_publisher.py
  src/robot_subscriber.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
```

### Step 4: Create a Service Package

```bash
cd ~/ros2_workspace_demo/src

# Create a robot services package
ros2 pkg create --name robot_services --dependencies rclpy std_msgs robot_msgs example_interfaces

# Navigate to the package
cd robot_services
```

Edit the `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robot_services</name>
  <version>0.0.0</version>
  <description>Robot services package</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>robot_msgs</depend>
  <depend>example_interfaces</depend>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create the source directory and service files:

```bash
mkdir -p src
```

Create `src/task_service_server.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from robot_msgs.srv import TaskService
import time

class TaskServiceServer(Node):
    def __init__(self):
        super().__init__('task_service_server')

        # Create service
        self.srv = self.create_service(TaskService, 'execute_task', self.execute_task_callback)

        self.get_logger().info('Task Service Server initialized')

    def execute_task_callback(self, request, response):
        self.get_logger().info(f'Received task request: {request.task_name} with priority {request.priority}')

        # Simulate task execution
        time.sleep(2)  # Simulate processing time

        # In a real system, this would execute the actual task
        success = True  # Simulate successful execution
        message = f'Task {request.task_name} completed successfully'

        response.success = success
        response.message = message

        self.get_logger().info(f'Task execution result: {response.message}')
        return response

def main(args=None):
    rclpy.init(args=args)
    server = TaskServiceServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Create `src/task_service_client.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from robot_msgs.srv import TaskService

class TaskServiceClient(Node):
    def __init__(self):
        super().__init__('task_service_client')

        # Create client
        self.cli = self.create_client(TaskService, 'execute_task')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Task service not available, waiting again...')

        self.get_logger().info('Task Service Client initialized')

    def send_task_request(self, task_name, priority):
        request = TaskService.Request()
        request.task_name = task_name
        request.priority = priority

        self.get_logger().info(f'Sending task request: {task_name} with priority {priority}')

        future = self.cli.call_async(request)
        return future

def main(args=None):
    rclpy.init(args=args)
    client = TaskServiceClient()

    # Send a few task requests
    tasks = [
        ("Navigation", 1),
        ("ObjectDetection", 2),
        ("Manipulation", 3)
    ]

    for task_name, priority in tasks:
        future = client.send_task_request(task_name, priority)

        # Wait for response
        rclpy.spin_until_future_complete(client, future)

        try:
            response = future.result()
            if response.success:
                client.get_logger().info(f'Success: {response.message}')
            else:
                client.get_logger().info(f'Failed: {response.message}')
        except Exception as e:
            client.get_logger().error(f'Service call failed: {e}')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Update the `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(robot_services)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(robot_msgs REQUIRED)
find_package(example_interfaces REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  src/task_service_server.py
  src/task_service_client.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
```

### Step 5: Build the Workspace

```bash
cd ~/ros2_workspace_demo

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Build the workspace
colcon build --packages-select robot_msgs robot_controller robot_services

# Source the built workspace
source install/setup.bash
```

Expected output: All packages should build successfully without errors.

### Common Issues

- **Package dependencies not found**: Ensure all dependencies are properly declared in package.xml
- **Build errors**: Check CMakeLists.txt and package.xml for syntax errors
- **Import errors**: Make sure to source the workspace after building

## Practical Example

Now let's create a launch file to bring up the entire system:

```bash
# Create a launch package
cd ~/ros2_workspace_demo/src
ros2 pkg create --name robot_launch --dependencies launch_ros

cd robot_launch
```

Edit `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robot_launch</name>
  <version>0.0.0</version>
  <description>Launch files for robot system</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>launch_ros</depend>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
</package>
```

Create `launch/robot_system.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),

        # Robot publisher node
        Node(
            package='robot_controller',
            executable='robot_publisher.py',
            name='robot_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        ),

        # Robot subscriber node
        Node(
            package='robot_controller',
            executable='robot_subscriber.py',
            name='robot_subscriber',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        ),

        # Task service server
        Node(
            package='robot_services',
            executable='task_service_server.py',
            name='task_service_server',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        )
    ])
```

Update the `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(robot_launch)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(launch_ros REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

Now build the launch package:

```bash
cd ~/ros2_workspace_demo

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Build the launch package
colcon build --packages-select robot_launch

# Source the workspace
source install/setup.bash
```

Test the complete system:

```bash
# Terminal 1: Launch the system
source ~/ros2_workspace_demo/install/setup.bash
ros2 launch robot_launch robot_system.launch.py

# Terminal 2: Run the service client
source ~/ros2_workspace_demo/install/setup.bash
ros2 run robot_services task_service_client.py
```

Expected results: The system should start with all nodes communicating properly, and the service client should be able to call the task service successfully.

## Troubleshooting

### Common Error 1: Package Dependencies Not Found
**Cause**: Missing or incorrect dependency declarations in package.xml
**Solution**: Verify all dependencies are listed in both build_depend and exec_depend sections
**Prevention Tips**: Use ros2 pkg create to ensure proper template structure

### Common Error 2: Build System Issues
**Cause**: Incorrect CMakeLists.txt configuration
**Solution**: Follow standard ROS 2 CMakeLists.txt patterns and verify syntax
**Prevention Tips**: Use the ROS 2 documentation templates as reference

### Common Error 3: Import Errors After Build
**Cause**: Workspace not sourced after building
**Solution**: Always source install/setup.bash after building
**Prevention Tips**: Add sourcing to your .bashrc file for automatic loading

## Key Takeaways

- ROS 2 workspaces provide organization for multiple related packages
- Proper package structure ensures code reusability and maintainability
- The colcon build system efficiently handles multi-package builds
- Launch files enable coordinated system startup
- Custom messages, services, and actions extend ROS 2 capabilities

## Additional Resources

- [ROS 2 Workspaces Documentation](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html)
- [ROS 2 Package Creation](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html)
- [Colcon Build Tool](https://colcon.readthedocs.io/en/released/)
- [ROS 2 Launch Files](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Creating-Launch-Files.html)

## Self-Assessment

1. What is the difference between build_depend and exec_depend in package.xml?
2. How do you create custom message types in ROS 2?
3. What is the purpose of the colcon build system?
4. How do you create and use launch files to coordinate multiple nodes?
5. What are the advantages of using packages over standalone nodes?

<ChapterNavigation
  previous={{
    permalink: '/docs/module-01-ros2/chapter-01-02-nodes-topics-services',
    title: '1.2 Nodes, Topics, Services and Actions'
  }}
  next={{
    permalink: '/docs/module-01-ros2/chapter-01-04-urdf-robot-description',
    title: '1.4 URDF & Robot Description'
  }}
/>