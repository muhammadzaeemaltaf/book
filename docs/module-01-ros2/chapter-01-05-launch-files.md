---
id: chapter-01-05-launch-files
title: "Launch Files - Coordinating Complex Robot Systems"
sidebar_label: "Chapter 5: Launch Files - Coordinating Complex Robot Systems"
description: "Understanding the structure and syntax of ROS 2 launch files for multi-node robot systems"
keywords:
  - ROS 2
  - Launch Files
  - System Coordination
  - Multi-node Systems
prerequisites:
  - chapter-01-01-architecture
  - chapter-01-02-nodes-topics-services
  - chapter-01-03-workspaces-packages
---


# Chapter 5 - Launch Files - Coordinating Complex Robot Systems

## Learning Objectives

- Understand the structure and syntax of ROS 2 launch files
- Create launch files for multi-node robot systems
- Use launch arguments and parameters effectively
- Implement conditional node launching and grouping
- Debug common launch file issues

## Prerequisites

- Understanding of ROS 2 nodes, packages, and workspaces
- Completed Chapter 3: Building ROS 2 Packages and Workspaces
- Ubuntu 22.04 LTS with ROS 2 Humble installed
- Basic Python programming knowledge
- Familiarity with YAML syntax

## Introduction

Launch files in ROS 2 provide a powerful mechanism to start multiple nodes with specific configurations simultaneously. They allow you to define complex robot systems in a single file, making it easy to reproduce, test, and deploy your robotic applications.

Launch files go beyond simple node startup by enabling parameter configuration, conditional execution, and complex system orchestration. This chapter will guide you through creating sophisticated launch files for real-world robotic systems.

### Why Launch Files Are Essential

Launch files eliminate the need to manually start multiple nodes in separate terminals. They ensure consistent system startup, proper parameter configuration, and coordinated node lifecycle management.

### Real-world Applications

- Autonomous vehicle systems with perception, planning, and control nodes
- Manufacturing robots with multiple sensors and actuators
- Research robots with configurable experimental setups
- Simulation environments with multiple robots

### What You'll Build by the End

By completing this chapter, you will create:
- A comprehensive launch file for a multi-node robot system
- Parameterized launch files with configurable settings
- Conditional launch configurations for different scenarios
- A launch file that demonstrates advanced features like groups and remapping

## Core Concepts

### Launch Description

A launch description is a Python-based specification that defines what should be launched. It includes nodes, parameters, and launch actions that control the startup process.

### Launch Arguments

Launch arguments allow runtime configuration of launch files, enabling the same launch file to be used with different settings without modification.

### Node Groups and Namespaces

Groups organize nodes logically and can apply common settings. Namespaces provide name isolation for nodes operating in the same system.

### Event Handling

Launch files can respond to events like node startup, shutdown, and failure, enabling robust system management.

## Hands-On Tutorial

### Step 1: Create a Basic Launch File

First, let's create a simple launch file to understand the basic structure:

```python
# ~/ros2_workspace_demo/src/basic_launch/launch/simple_robot.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution

def generate_launch_description():
    # Define launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Create the launch description
    ld = LaunchDescription()

    # Add launch argument declaration
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    ))

    # Add nodes to the launch description
    ld.add_action(Node(
        package='robot_controller',
        executable='robot_publisher.py',
        name='robot_publisher',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    ))

    ld.add_action(Node(
        package='robot_controller',
        executable='robot_subscriber.py',
        name='robot_subscriber',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    ))

    return ld
```

Create the basic launch package:

```bash
cd ~/ros2_workspace_demo/src
ros2 pkg create --name basic_launch --dependencies launch_ros
cd basic_launch
```

Edit `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>basic_launch</name>
  <version>0.0.0</version>
  <description>Basic launch files for robot system</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>launch_ros</depend>
  <depend>robot_controller</depend>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create the launch directory and file:

```bash
mkdir launch
```

Create the launch file as shown above, then update `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(basic_launch)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(launch_ros REQUIRED)
find_package(robot_controller REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

### Step 2: Create a Parameterized Launch File

Now let's create a more sophisticated launch file with parameters:

```python
# ~/ros2_workspace_demo/src/advanced_launch/launch/parameterized_robot.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Define launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_name = LaunchConfiguration('robot_name', default='robot1')
    namespace = LaunchConfiguration('namespace', default='')
    config_file = LaunchConfiguration('config_file', default='')

    # Create the launch description
    ld = LaunchDescription()

    # Add launch argument declarations
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    ))

    ld.add_action(DeclareLaunchArgument(
        'robot_name',
        default_value='robot1',
        description='Name of the robot'
    ))

    ld.add_action(DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for the robot nodes'
    ))

    ld.add_action(DeclareLaunchArgument(
        'config_file',
        default_value='',
        description='Path to configuration file'
    ))

    # Create a group action with namespace if specified
    group_action = GroupAction(
        actions=[
            # Add a log message
            LogInfo(msg=['Starting robot: ', robot_name]),

            # Robot publisher with parameters
            Node(
                package='robot_controller',
                executable='robot_publisher.py',
                name='robot_publisher',
                namespace=namespace,
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_name': robot_name},
                ],
                remappings=[
                    ('robot_status', 'status'),
                    ('robot_commands', 'commands'),
                ],
                output='screen'
            ),

            # Robot subscriber with parameters
            Node(
                package='robot_controller',
                executable='robot_subscriber.py',
                name='robot_subscriber',
                namespace=namespace,
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_name': robot_name},
                ],
                output='screen'
            ),

            # Task service server
            Node(
                package='robot_services',
                executable='task_service_server.py',
                name='task_service_server',
                namespace=namespace,
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_name': robot_name},
                ],
                output='screen'
            ),
        ]
    )

    ld.add_action(group_action)

    return ld
```

Create the advanced launch package:

```bash
cd ~/ros2_workspace_demo/src
ros2 pkg create --name advanced_launch --dependencies launch_ros robot_controller robot_services
cd advanced_launch
```

Edit `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>advanced_launch</name>
  <version>0.0.0</version>
  <description>Advanced launch files for robot system</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>launch_ros</depend>
  <depend>robot_controller</depend>
  <depend>robot_services</depend>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create the launch directory and file, then update `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(advanced_launch)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(launch_ros REQUIRED)
find_package(robot_controller REQUIRED)
find_package(robot_services REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

### Step 3: Create a Complex Multi-Robot Launch File

Now let's create a launch file that can start multiple robots:

```python
# ~/ros2_workspace_demo/src/multirobot_launch/launch/multi_robot_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch.conditions import IfCondition

def generate_launch_description():
    # Define launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    enable_robot1 = LaunchConfiguration('enable_robot1', default='true')
    enable_robot2 = LaunchConfiguration('enable_robot2', default='true')
    enable_robot3 = LaunchConfiguration('enable_robot3', default='false')
    robot_count = LaunchConfiguration('robot_count', default='2')

    # Create the launch description
    ld = LaunchDescription()

    # Add launch argument declarations
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    ))

    ld.add_action(DeclareLaunchArgument(
        'enable_robot1',
        default_value='true',
        description='Enable robot 1'
    ))

    ld.add_action(DeclareLaunchArgument(
        'enable_robot2',
        default_value='true',
        description='Enable robot 2'
    ))

    ld.add_action(DeclareLaunchArgument(
        'enable_robot3',
        default_value='false',
        description='Enable robot 3'
    ))

    ld.add_action(DeclareLaunchArgument(
        'robot_count',
        default_value='2',
        description='Number of robots to start'
    ))

    # Add a log message
    ld.add_action(LogInfo(
        msg=['Starting multi-robot system with ', robot_count, ' robots']
    ))

    # Robot 1 group
    robot1_group = GroupAction(
        condition=IfCondition(enable_robot1),
        actions=[
            PushRosNamespace('robot1'),
            Node(
                package='robot_controller',
                executable='robot_publisher.py',
                name='robot_publisher',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_name': 'robot1'},
                ],
                output='screen'
            ),
            Node(
                package='robot_controller',
                executable='robot_subscriber.py',
                name='robot_subscriber',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_name': 'robot1'},
                ],
                output='screen'
            ),
        ]
    )
    ld.add_action(robot1_group)

    # Robot 2 group
    robot2_group = GroupAction(
        condition=IfCondition(enable_robot2),
        actions=[
            PushRosNamespace('robot2'),
            Node(
                package='robot_controller',
                executable='robot_publisher.py',
                name='robot_publisher',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_name': 'robot2'},
                ],
                output='screen'
            ),
            Node(
                package='robot_controller',
                executable='robot_subscriber.py',
                name='robot_subscriber',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_name': 'robot2'},
                ],
                output='screen'
            ),
        ]
    )
    ld.add_action(robot2_group)

    # Robot 3 group (conditionally enabled)
    robot3_group = GroupAction(
        condition=IfCondition(enable_robot3),
        actions=[
            PushRosNamespace('robot3'),
            Node(
                package='robot_controller',
                executable='robot_publisher.py',
                name='robot_publisher',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_name': 'robot3'},
                ],
                output='screen'
            ),
            Node(
                package='robot_controller',
                executable='robot_subscriber.py',
                name='robot_subscriber',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'robot_name': 'robot3'},
                ],
                output='screen'
            ),
        ]
    )
    ld.add_action(robot3_group)

    # Central coordinator node
    ld.add_action(Node(
        package='robot_services',
        executable='task_service_server.py',
        name='central_coordinator',
        parameters=[
            {'use_sim_time': use_sim_time},
        ],
        output='screen'
    ))

    return ld
```

Create the multirobot launch package:

```bash
cd ~/ros2_workspace_demo/src
ros2 pkg create --name multirobot_launch --dependencies launch_ros robot_controller robot_services
cd multirobot_launch
```

Edit `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>multirobot_launch</name>
  <version>0.0.0</version>
  <description>Multi-robot launch files for robot system</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>launch_ros</depend>
  <depend>robot_controller</depend>
  <depend>robot_services</depend>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create the launch directory and file, then update `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(multirobot_launch)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(launch_ros REQUIRED)
find_package(robot_controller REQUIRED)
find_package(robot_services REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

### Step 4: Create a Launch File with Event Handling

Let's create a launch file that demonstrates event handling:

```python
# ~/ros2_workspace_demo/src/event_launch/launch/event_driven_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessStart, OnProcessExit, OnProcessIO
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Define launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Create the launch description
    ld = LaunchDescription()

    # Add launch argument declarations
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    ))

    # Create a robot publisher node
    robot_publisher_node = Node(
        package='robot_controller',
        executable='robot_publisher.py',
        name='robot_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
        ],
        output='screen'
    )

    # Create a robot subscriber node
    robot_subscriber_node = Node(
        package='robot_controller',
        executable='robot_subscriber.py',
        name='robot_subscriber',
        parameters=[
            {'use_sim_time': use_sim_time},
        ],
        output='screen'
    )

    # Register event handlers
    # Log when publisher starts
    publisher_start_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_publisher_node,
            on_start=[
                LogInfo(msg='Robot publisher has started successfully')
            ]
        )
    )

    # Log when subscriber starts
    subscriber_start_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_subscriber_node,
            on_start=[
                LogInfo(msg='Robot subscriber has started successfully')
            ]
        )
    )

    # Log when either node exits
    publisher_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=robot_publisher_node,
            on_exit=[
                LogInfo(msg='Robot publisher has exited')
            ]
        )
    )

    subscriber_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=robot_subscriber_node,
            on_exit=[
                LogInfo(msg='Robot subscriber has exited')
            ]
        )
    )

    # Add all actions to the launch description
    ld.add_action(publisher_start_handler)
    ld.add_action(subscriber_start_handler)
    ld.add_action(publisher_exit_handler)
    ld.add_action(subscriber_exit_handler)
    ld.add_action(robot_publisher_node)
    ld.add_action(robot_subscriber_node)

    return ld
```

Create the event launch package:

```bash
cd ~/ros2_workspace_demo/src
ros2 pkg create --name event_launch --dependencies launch_ros robot_controller
cd event_launch
```

Edit `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>event_launch</name>
  <version>0.0.0</version>
  <description>Event-driven launch files for robot system</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>launch_ros</depend>
  <depend>robot_controller</depend>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create the launch directory and file, then update `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(event_launch)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(launch_ros REQUIRED)
find_package(robot_controller REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

### Step 5: Build All New Packages

```bash
cd ~/ros2_workspace_demo

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Build all new packages
colcon build --packages-select basic_launch advanced_launch multirobot_launch event_launch

# Source the workspace
source install/setup.bash
```

### Step 6: Test the Launch Files

Test the basic launch file:

```bash
# Terminal 1: Launch basic robot system
source ~/ros2_workspace_demo/install/setup.bash
ros2 launch basic_launch simple_robot.launch.py use_sim_time:=false
```

Test the parameterized launch file:

```bash
# Terminal 1: Launch parameterized robot system
source ~/ros2_workspace_demo/install/setup.bash
ros2 launch advanced_launch parameterized_robot.launch.py robot_name:=my_robot namespace:=my_namespace
```

Test the multi-robot launch file:

```bash
# Terminal 1: Launch multi-robot system
source ~/ros2_workspace_demo/install/setup.bash
ros2 launch multirobot_launch multi_robot_system.launch.py enable_robot1:=true enable_robot2:=true enable_robot3:=false
```

Test the event-driven launch file:

```bash
# Terminal 1: Launch event-driven system
source ~/ros2_workspace_demo/install/setup.bash
ros2 launch event_launch event_driven_system.launch.py
```

Expected output: All launch files should start the specified nodes with appropriate logging and event handling.

### Common Issues

- **Launch file not found**: Ensure launch files are in the correct directory and properly installed
- **Node not found**: Verify that the package containing the node executable is built and sourced
- **Parameter issues**: Check parameter syntax and ensure they're properly defined in the node

## Practical Example

Now let's create a comprehensive launch file that demonstrates all the concepts together:

```python
# ~/ros2_workspace_demo/src/comprehensive_launch/launch/comprehensive_robot_system.launch.py
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    LogInfo,
    RegisterEventHandler
)
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    # Define launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    enable_simulation = LaunchConfiguration('enable_simulation', default='false')
    robot_count = LaunchConfiguration('robot_count', default='2')
    enable_logging = LaunchConfiguration('enable_logging', default='true')

    # Create the launch description
    ld = LaunchDescription()

    # Add launch argument declarations
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    ))

    ld.add_action(DeclareLaunchArgument(
        'enable_simulation',
        default_value='false',
        description='Enable simulation-specific configurations'
    ))

    ld.add_action(DeclareLaunchArgument(
        'robot_count',
        default_value='2',
        description='Number of robots to start (1-3)'
    ))

    ld.add_action(DeclareLaunchArgument(
        'enable_logging',
        default_value='true',
        description='Enable detailed logging'
    ))

    # Add system startup log
    ld.add_action(LogInfo(
        msg=['Starting comprehensive robot system with ', robot_count, ' robots']
    ))

    # Create robot groups based on robot_count
    for i in range(1, 4):  # Support up to 3 robots
        robot_group = GroupAction(
            condition=IfCondition(f'$(eval {i} <= int({robot_count}))'),
            actions=[
                PushRosNamespace(f'robot{i}'),

                # Robot publisher with event handling
                Node(
                    package='robot_controller',
                    executable='robot_publisher.py',
                    name='robot_publisher',
                    parameters=[
                        {'use_sim_time': use_sim_time},
                        {'robot_name': f'robot{i}'},
                        {'simulation_mode': enable_simulation},
                    ],
                    output='screen',
                    respawn=True,  # Restart if it crashes
                    respawn_delay=2.0
                ),

                # Robot subscriber with event handling
                Node(
                    package='robot_controller',
                    executable='robot_subscriber.py',
                    name='robot_subscriber',
                    parameters=[
                        {'use_sim_time': use_sim_time},
                        {'robot_name': f'robot{i}'},
                        {'simulation_mode': enable_simulation},
                    ],
                    output='screen'
                ),

                # Conditional simulation node
                Node(
                    package='robot_services',
                    executable='task_service_server.py',
                    name='task_service_server',
                    parameters=[
                        {'use_sim_time': use_sim_time},
                        {'robot_name': f'robot{i}'},
                    ],
                    output='screen',
                    condition=UnlessCondition(enable_simulation)
                ),
            ]
        )
        ld.add_action(robot_group)

    # Central coordinator node
    coordinator_node = Node(
        package='robot_services',
        executable='task_service_server.py',
        name='central_coordinator',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'system_mode': 'multi_robot'},
        ],
        output='screen'
    )

    # Add coordinator node and event handlers
    ld.add_action(coordinator_node)

    # Event handlers for coordinator
    ld.add_action(RegisterEventHandler(
        OnProcessStart(
            target_action=coordinator_node,
            on_start=[
                LogInfo(msg='Central coordinator has started successfully')
            ]
        )
    ))

    ld.add_action(RegisterEventHandler(
        OnProcessExit(
            target_action=coordinator_node,
            on_exit=[
                LogInfo(msg='Central coordinator has exited - shutting down system')
            ]
        )
    ))

    # Add final log
    ld.add_action(LogInfo(
        msg=['Comprehensive robot system launch configuration complete']
    ))

    return ld
```

Create the comprehensive launch package:

```bash
cd ~/ros2_workspace_demo/src
ros2 pkg create --name comprehensive_launch --dependencies launch_ros robot_controller robot_services
cd comprehensive_launch
```

Edit `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>comprehensive_launch</name>
  <version>0.0.0</version>
  <description>Comprehensive launch files for robot system</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>launch_ros</depend>
  <depend>robot_controller</depend>
  <depend>robot_services</depend>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Create the launch directory and file, then update `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(comprehensive_launch)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(launch_ros REQUIRED)
find_package(robot_controller REQUIRED)
find_package(robot_services REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

Build the comprehensive launch package:

```bash
cd ~/ros2_workspace_demo

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Build the comprehensive launch package
colcon build --packages-select comprehensive_launch

# Source the workspace
source install/setup.bash
```

Test the comprehensive launch file:

```bash
# Terminal 1: Launch comprehensive system
source ~/ros2_workspace_demo/install/setup.bash
ros2 launch comprehensive_launch comprehensive_robot_system.launch.py robot_count:=2 use_sim_time:=false
```

Expected results: The system should start 2 robots with proper namespacing, event handling, and conditional logic based on the parameters.

## Troubleshooting

### Common Error 1: Launch File Syntax Errors
**Cause**: Incorrect Python syntax or missing imports in launch file
**Solution**: Verify all required imports and Python syntax is correct
**Prevention Tips**: Use proper Python linting and follow ROS 2 launch file templates

### Common Error 2: Parameter Not Found
**Cause**: Parameter name mismatch between launch file and node
**Solution**: Verify parameter names match exactly between both files
**Prevention Tips**: Use consistent naming conventions and document parameters

### Common Error 3: Node Executable Not Found
**Cause**: Node executable not built or not in the correct location
**Solution**: Ensure package is built and executable is properly installed
**Prevention Tips**: Use the correct executable name and verify installation paths

## Key Takeaways

- Launch files enable coordinated startup of complex robotic systems
- Launch arguments provide runtime configurability without code changes
- Namespaces and groups organize nodes logically in multi-robot systems
- Event handlers enable robust system management and monitoring
- Conditional execution allows flexible system configurations

## Additional Resources

- [ROS 2 Launch System Documentation](https://docs.ros.org/en/humble/How-To-Guides/Launch-system.html)
- [Launch File Best Practices](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Creating-Launch-Files.html)
- [Launch Arguments and Parameters](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Using-Launch-Arguments.html)
- [Event Handling in Launch Files](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Handling-Events.html)

## Self-Assessment

1. What is the difference between launch arguments and parameters in ROS 2?
2. How do you create conditional node launching in a launch file?
3. What is the purpose of namespaces in multi-robot systems?
4. How do event handlers improve system reliability in launch files?
5. What are the advantages of using launch files over manual node startup?

<ChapterNavigation
  previous={{
    permalink: '/docs/module-01-ros2/chapter-01-04-urdf-robot-description',
    title: '1.4 URDF & Robot Description'
  }}
  next={{
    permalink: '/docs/module-02-digital-twin/chapter-02-01-simulation-fundamentals',
    title: '2.1 Simulation Fundamentals'
  }}
/>