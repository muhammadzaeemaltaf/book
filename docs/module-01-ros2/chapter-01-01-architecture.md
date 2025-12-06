---
id: chapter-01-01-architecture
title: ROS 2 Architecture & Core Concepts
sidebar_label: "Chapter 1: ROS 2 Architecture & Core Concepts"
description: Understanding the fundamental architecture of ROS 2 and its key components
keywords:
  - ROS 2
  - Architecture
  - Nodes
  - Topics
  - Services
prerequisites: []
---


# Chapter 1 - ROS 2 Architecture & Core Concepts

## Learning Objectives

- Understand the fundamental architecture of ROS 2
- Identify the key components of a ROS 2 system
- Explain the role of nodes, topics, and services in ROS 2
- Set up a basic ROS 2 workspace for development

## Prerequisites

- Completion of the introduction module
- Ubuntu 22.04 LTS with ROS 2 Humble Hawksbill installed
- Basic understanding of distributed systems
- Python programming knowledge

## Introduction

ROS 2 (Robot Operating System 2) provides the foundation for all robotic applications. Think of it as the nervous system of a robot - it allows sensors to communicate with controllers, enables coordination between different software components, and provides the infrastructure for complex robotic behaviors.

ROS 2 represents a significant evolution from ROS 1, addressing critical issues around security, real-time performance, and multi-robot systems. Its architecture is built on DDS (Data Distribution Service) which provides a robust, scalable communication layer.

### Why ROS 2 Architecture Matters

The ROS 2 architecture enables modular robot development where different teams can work on separate components that seamlessly integrate. This approach allows for rapid prototyping, testing, and deployment of robotic systems.

### Real-world Applications

- Autonomous vehicles coordinating sensor data and control commands
- Manufacturing robots with multiple arms working in coordination
- Drone swarms with distributed decision-making capabilities
- Service robots integrating perception, navigation, and manipulation

### What You'll Build by the End

By completing this chapter, you will create a simple ROS 2 system with:
- A publisher node that generates sensor data
- A subscriber node that processes the data
- A service client and server for on-demand operations
- A launch file to start the entire system

## Core Concepts

### DDS-Based Communication

ROS 2 uses DDS (Data Distribution Service) as its underlying communication middleware. DDS provides quality of service (QoS) policies that ensure reliable communication even in challenging environments.

### Nodes, Topics, and Services

The fundamental building blocks of ROS 2 enable flexible and robust robot architectures:
- **Nodes**: Independent processes that perform specific functions
- **Topics**: Asynchronous communication channels for continuous data streams
- **Services**: Synchronous request-response communication for on-demand operations

### Lifecycle Management

ROS 2 provides explicit lifecycle management for nodes, allowing for better resource management and coordinated system startup/shutdown.

## Hands-On Tutorial

### Step 1: Verify ROS 2 Installation

First, verify that ROS 2 Humble is properly installed and sourced:

```bash
# Source the ROS 2 installation
source /opt/ros/humble/setup.bash

# Verify ROS 2 is working
ros2 --version

# Check available commands
ros2 --help
```

Expected output: ROS 2 version information and command list should display without errors.

### Step 2: Create a ROS 2 Workspace

Create a dedicated workspace for your ROS 2 development:

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace (even though it's empty)
colcon build

# Source the workspace
source install/setup.bash
```

### Step 3: Create a Simple Publisher Node

Let's create a simple publisher that generates mock sensor data:

```python
# ~/ros2_ws/src/sensor_publisher.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher_ = self.create_publisher(String, 'sensor_data', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Sensor reading {self.i}: {random.uniform(0, 100):.2f}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = SensorPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Create a Simple Subscriber Node

Now create a subscriber that receives and processes the sensor data:

```python
# ~/ros2_ws/src/sensor_subscriber.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SensorSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_subscriber')
        self.subscription = self.create_subscription(
            String,
            'sensor_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    sensor_subscriber = SensorSubscriber()

    try:
        rclpy.spin(sensor_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 5: Test the Publisher-Subscriber System

```bash
# Terminal 1: Start the publisher
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/sensor_publisher.py

# Terminal 2: Start the subscriber
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/sensor_subscriber.py
```

Expected output: The subscriber should receive and display messages published by the publisher.

### Common Issues

- **ROS 2 not found**: Ensure you've sourced the correct ROS 2 installation
- **Permission errors**: Check file permissions for your Python scripts
- **Communication issues**: Verify both nodes are on the same network/DDS domain

## Practical Example

In this example, we'll create a more sophisticated ROS 2 system that demonstrates the architecture principles:

```python
# ~/ros2_ws/src/ros2_architecture_demo.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from example_interfaces.srv import AddTwoInts
import threading
import time

class DataProcessor(Node):
    """A node that processes incoming data and provides services"""

    def __init__(self):
        super().__init__('data_processor')

        # Publishers
        self.status_pub = self.create_publisher(String, 'system_status', 10)
        self.processed_data_pub = self.create_publisher(Int32, 'processed_data', 10)

        # Subscribers
        self.sensor_sub = self.create_subscription(
            String, 'raw_sensor_data', self.sensor_callback, 10)

        # Services
        self.add_srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_callback)

        # Timers
        self.status_timer = self.create_timer(1.0, self.publish_status)

        self.processed_count = 0
        self.get_logger().info('Data Processor node initialized')

    def sensor_callback(self, msg):
        """Process incoming sensor data"""
        try:
            # Extract numeric value from sensor message
            raw_data = msg.data
            # Simple processing: extract last number and square it
            import re
            numbers = re.findall(r'\d+\.?\d*', raw_data)
            if numbers:
                value = float(numbers[-1])
                processed_value = int(value ** 0.5)  # Square root, converted to int

                # Publish processed data
                processed_msg = Int32()
                processed_msg.data = processed_value
                self.processed_data_pub.publish(processed_msg)

                self.processed_count += 1
                self.get_logger().info(f'Processed: {value} -> {processed_value}')
        except Exception as e:
            self.get_logger().error(f'Error processing sensor data: {e}')

    def add_callback(self, request, response):
        """Service callback to add two integers"""
        response.sum = request.a + request.b
        self.get_logger().info(f'Adding {request.a} + {request.b} = {response.sum}')
        return response

    def publish_status(self):
        """Publish system status periodically"""
        status_msg = String()
        status_msg.data = f'Active: {rclpy.ok()}, Processed: {self.processed_count}'
        self.status_pub.publish(status_msg)

def run_publisher(node):
    """Simulate a sensor publisher in a separate thread"""
    import random

    rclpy.init()
    pub_node = rclpy.create_node('simulated_sensor')
    publisher = pub_node.create_publisher(String, 'raw_sensor_data', 10)

    while rclpy.ok():
        msg = String()
        msg.data = f'Sensor reading: {random.uniform(10, 90):.2f}'
        publisher.publish(msg)
        node.get_logger().info(f'Simulated sensor published: {msg.data}')
        time.sleep(2)

def main(args=None):
    rclpy.init(args=args)
    processor = DataProcessor()

    # Run the simulated sensor publisher in a separate thread
    publisher_thread = threading.Thread(target=run_publisher, args=(processor,))
    publisher_thread.daemon = True
    publisher_thread.start()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Shutting down...')
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Run this example to see the ROS 2 architecture in action:
```bash
python3 ~/ros2_ws/src/ros2_architecture_demo.py
```

Expected results: The system should demonstrate nodes communicating via topics and services, showing the core ROS 2 architecture.

## Troubleshooting

### Common Error 1: Node Communication Issues
**Cause**: Nodes not communicating due to different DDS domains or network issues
**Solution**: Ensure all nodes are using the same ROS_DOMAIN_ID and are on the same network
**Prevention Tips**: Use consistent environment setup across all terminals

### Common Error 2: Permission Denied for ROS Installation
**Cause**: ROS installed in system directories without proper permissions
**Solution**: Ensure ROS installation is properly set up with correct permissions
**Prevention Tips**: Follow official ROS installation guides carefully

### Common Error 3: Import Errors
**Cause**: Python modules not found or incorrect environment setup
**Solution**: Source ROS environment in each terminal session
**Prevention Tips**: Add sourcing commands to your .bashrc file

## Key Takeaways

- ROS 2 architecture provides a flexible, distributed system for robotics
- Nodes communicate via topics (asynchronous) and services (synchronous)
- DDS-based communication ensures reliable data distribution
- Proper workspace setup is essential for ROS 2 development
- Quality of Service (QoS) policies enable reliable communication in real-world scenarios

## Additional Resources

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [DDS Primer](https://www.dds-foundation.org/what-is-dds-technology/)
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [Quality of Service in ROS 2](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html)

## Self-Assessment

1. What is the primary difference between ROS 1 and ROS 2 architecture?
2. Explain the role of DDS in ROS 2 communication.
3. When would you use a topic versus a service in ROS 2?
4. What are Quality of Service (QoS) policies and why are they important?
5. How do you create a ROS 2 workspace and what is its purpose?

<ChapterNavigation
  previous={{
    permalink: '/docs/module-01-ros2/index',
    title: 'Module 1: ROS 2'
  }}
  next={{
    permalink: '/docs/module-01-ros2/chapter-01-02-nodes-topics-services',
    title: '1.2 Nodes, Topics, Services and Actions'
  }}
/>