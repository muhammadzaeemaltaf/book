---
id: chapter-01-02-nodes-topics-services
title: Nodes, Topics, Services and Actions
sidebar_label: Nodes, Topics, Services and Actions
description: Understanding ROS 2 communication patterns including nodes, topics, services, and actions
keywords:
  - ROS 2
  - Nodes
  - Topics
  - Services
  - Actions
  - Communication
prerequisites:
  - chapter-01-01-architecture
---


# Nodes, Topics, and Services in Depth

## Learning Objectives

- Master the creation and management of ROS 2 nodes
- Implement publisher-subscriber communication patterns
- Design and implement service-based interactions
- Understand Quality of Service (QoS) settings for reliable communication
- Debug common communication issues in ROS 2 systems

## Prerequisites

- Understanding of ROS 2 architecture fundamentals
- Completed Chapter 1: ROS 2 Architecture & Core Concepts
- Ubuntu 22.04 LTS with ROS 2 Humble installed
- Basic Python programming skills
- Familiarity with terminal operations

## Introduction

Nodes, topics, and services form the fundamental communication infrastructure of ROS 2. These three concepts enable the distributed nature of robotic systems, allowing different components to interact seamlessly regardless of their physical location or implementation language.

Nodes are the basic execution units that perform specific functions. Topics enable asynchronous, one-to-many communication through a publish-subscribe pattern, while services provide synchronous request-response communication for on-demand operations.

### Why Master Communication Patterns

Understanding these communication patterns is crucial for designing efficient and maintainable robotic systems. The choice between topics and services affects system performance, reliability, and real-time behavior.

### Real-world Applications

- Sensor fusion systems combining data from multiple sources
- Navigation systems with path planning and obstacle avoidance
- Multi-robot coordination and communication
- Human-robot interaction interfaces

### What You'll Build by the End

By completing this chapter, you will create a complete ROS 2 communication system with:
- Multiple publisher nodes sending different types of data
- Subscriber nodes that process and aggregate information
- Service nodes providing on-demand calculations and operations
- A launch file to coordinate the entire system

## Core Concepts

### Nodes: The Execution Units

Nodes are processes that perform computation. In ROS 2, nodes are designed to be modular and reusable, each responsible for a specific task within the larger robotic system.

### Topics: Asynchronous Communication

Topics enable asynchronous communication using a publish-subscribe pattern. Publishers send messages to topics without knowing who subscribes, and subscribers receive messages without knowing who publishes.

### Services: Synchronous Communication

Services provide synchronous request-response communication. A client sends a request and waits for a response, making services suitable for on-demand operations.

### Quality of Service (QoS)

QoS settings allow fine-tuning of communication behavior, including reliability, durability, and history policies.

## Hands-On Tutorial

### Step 1: Create a Node with Multiple Publishers

Let's create a comprehensive node that publishes multiple types of data:

```python
# ~/ros2_ws/src/multi_publisher_node.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32
from geometry_msgs.msg import Twist
import random
import math

class MultiPublisherNode(Node):
    def __init__(self):
        super().__init__('multi_publisher_node')

        # Create publishers for different data types
        self.sensor_pub = self.create_publisher(String, 'sensor_data', 10)
        self.counter_pub = self.create_publisher(Int32, 'counter', 10)
        self.temperature_pub = self.create_publisher(Float32, 'temperature', 10)
        self.velocity_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Timer to publish data periodically
        self.timer = self.create_timer(0.5, self.publish_data)
        self.counter = 0

        self.get_logger().info('Multi-Publisher Node initialized')

    def publish_data(self):
        # Publish sensor data
        sensor_msg = String()
        sensor_msg.data = f'Sensor reading {self.counter}: {random.uniform(20, 40):.2f}°C'
        self.sensor_pub.publish(sensor_msg)

        # Publish counter
        counter_msg = Int32()
        counter_msg.data = self.counter
        self.counter_pub.publish(counter_msg)

        # Publish temperature
        temp_msg = Float32()
        temp_msg.data = random.uniform(15, 35)
        self.temperature_pub.publish(temp_msg)

        # Publish velocity command (simulating robot movement)
        vel_msg = Twist()
        vel_msg.linear.x = random.uniform(0.1, 0.5)  # Forward velocity
        vel_msg.angular.z = random.uniform(-0.5, 0.5)  # Angular velocity
        self.velocity_pub.publish(vel_msg)

        self.counter += 1
        self.get_logger().info(f'Published data set {self.counter}')

def main(args=None):
    rclpy.init(args=args)
    node = MultiPublisherNode()

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

### Step 2: Create a Multi-Subscriber Node

Now create a node that subscribes to multiple topics:

```python
# ~/ros2_ws/src/multi_subscriber_node.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32
from geometry_msgs.msg import Twist

class MultiSubscriberNode(Node):
    def __init__(self):
        super().__init__('multi_subscriber_node')

        # Create subscribers for different topics
        self.sensor_sub = self.create_subscription(
            String, 'sensor_data', self.sensor_callback, 10)
        self.counter_sub = self.create_subscription(
            Int32, 'counter', self.counter_callback, 10)
        self.temperature_sub = self.create_subscription(
            Float32, 'temperature', self.temperature_callback, 10)
        self.velocity_sub = self.create_subscription(
            Twist, 'cmd_vel', self.velocity_callback, 10)

        # Store latest values
        self.latest_sensor = "No data"
        self.latest_counter = 0
        self.latest_temperature = 0.0

        self.get_logger().info('Multi-Subscriber Node initialized')

    def sensor_callback(self, msg):
        self.latest_sensor = msg.data
        self.get_logger().info(f'Sensor: {msg.data}')
        self.log_aggregated_data()

    def counter_callback(self, msg):
        self.latest_counter = msg.data
        self.get_logger().info(f'Counter: {msg.data}')
        self.log_aggregated_data()

    def temperature_callback(self, msg):
        self.latest_temperature = msg.data
        self.get_logger().info(f'Temperature: {msg.data}°C')
        self.log_aggregated_data()

    def velocity_callback(self, msg):
        self.get_logger().info(f'Velocity: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}')
        self.log_aggregated_data()

    def log_aggregated_data(self):
        """Log a summary of all received data"""
        summary = f"Aggregated - Counter: {self.latest_counter}, Temp: {self.latest_temperature:.2f}°C, Sensor: {self.latest_sensor}"
        self.get_logger().info(summary)

def main(args=None):
    rclpy.init(args=args)
    node = MultiSubscriberNode()

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

### Step 3: Create a Service Server

Create a service that performs calculations based on the data:

```python
# ~/ros2_ws/src/data_analysis_service.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger, AddTwoInts
from std_msgs.msg import Float32
import statistics

class DataAnalysisService(Node):
    def __init__(self):
        super().__init__('data_analysis_service')

        # Create services
        self.stats_service = self.create_service(
            Trigger, 'calculate_statistics', self.calculate_statistics_callback)
        self.add_service = self.create_service(
            AddTwoInts, 'add_values', self.add_values_callback)

        # Store temperature history for statistics
        self.temperature_history = []

        # Subscriber to collect temperature data
        self.temp_sub = self.create_subscription(
            Float32, 'temperature', self.temperature_collector, 10)

        self.get_logger().info('Data Analysis Service initialized')

    def temperature_collector(self, msg):
        """Collect temperature data for statistics"""
        self.temperature_history.append(msg.data)
        # Keep only the last 50 readings to avoid memory issues
        if len(self.temperature_history) > 50:
            self.temperature_history.pop(0)

    def calculate_statistics_callback(self, request, response):
        """Calculate statistics on collected temperature data"""
        if not self.temperature_history:
            response.success = False
            response.message = "No temperature data available"
            return response

        avg_temp = statistics.mean(self.temperature_history)
        min_temp = min(self.temperature_history)
        max_temp = max(self.temperature_history)
        std_dev = statistics.stdev(self.temperature_history) if len(self.temperature_history) > 1 else 0.0

        stats_msg = f"Temperature Stats: Avg={avg_temp:.2f}°C, Min={min_temp:.2f}°C, Max={max_temp:.2f}°C, StdDev={std_dev:.2f}"

        response.success = True
        response.message = stats_msg
        self.get_logger().info(f'Statistics calculated: {stats_msg}')

        return response

    def add_values_callback(self, request, response):
        """Add two integer values"""
        response.sum = request.a + request.b
        self.get_logger().info(f'Added {request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = DataAnalysisService()

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

### Step 4: Create a Service Client

Create a client that uses the services:

```python
# ~/ros2_ws/src/service_client.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger, AddTwoInts
import time

class ServiceClient(Node):
    def __init__(self):
        super().__init__('service_client')

        # Create clients
        self.stats_client = self.create_client(Trigger, 'calculate_statistics')
        self.add_client = self.create_client(AddTwoInts, 'add_values')

        # Wait for services to be available
        while not self.stats_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Statistics service not available, waiting again...')

        while not self.add_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Add service not available, waiting again...')

        self.get_logger().info('Service clients initialized')

    def call_statistics_service(self):
        """Call the statistics service"""
        request = Trigger.Request()
        future = self.stats_client.call_async(request)

        # Wait for response
        rclpy.spin_until_future_complete(self, future)

        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Statistics: {response.message}')
            else:
                self.get_logger().info(f'Statistics request failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def call_add_service(self, a, b):
        """Call the add service"""
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        future = self.add_client.call_async(request)

        # Wait for response
        rclpy.spin_until_future_complete(self, future)

        try:
            response = future.result()
            self.get_logger().info(f'{a} + {b} = {response.sum}')
            return response.sum
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return None

def main(args=None):
    rclpy.init(args=args)
    client = ServiceClient()

    # Call services periodically
    for i in range(5):
        client.call_statistics_service()
        client.call_add_service(i * 10, (i + 1) * 5)
        time.sleep(2)

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 5: Test the Complete System

```bash
# Terminal 1: Start the publisher
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/multi_publisher_node.py

# Terminal 2: Start the subscriber
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/multi_subscriber_node.py

# Terminal 3: Start the service server
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/data_analysis_service.py

# Terminal 4: Start the service client
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/service_client.py
```

Expected output: You should see all nodes communicating with each other, with data flowing from publishers to subscribers and services being called and responding appropriately.

### Common Issues

- **Service not available**: Ensure the service server is running before the client
- **Topic mismatch**: Verify topic names match between publishers and subscribers
- **Node name conflicts**: Use unique node names to avoid conflicts

## Practical Example

In this comprehensive example, we'll create a complete ROS 2 system that demonstrates all communication patterns:

```python
# ~/ros2_ws/src/ros2_communication_demo.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32
from geometry_msgs.msg import Twist
from example_interfaces.srv import Trigger, AddTwoInts
from example_interfaces.action import Fibonacci
import rclpy.action
import threading
import time
import random

class CommunicationDemoNode(Node):
    """A comprehensive ROS 2 communication demo node"""

    def __init__(self):
        super().__init__('communication_demo_node')

        # Publishers
        self.status_pub = self.create_publisher(String, 'system_status', 10)
        self.sensor_pub = self.create_publisher(Float32, 'sensor_readings', 10)
        self.nav_cmd_pub = self.create_publisher(Twist, 'navigation_commands', 10)

        # Subscribers
        self.status_sub = self.create_subscription(
            String, 'system_status', self.status_callback, 10)
        self.sensor_sub = self.create_subscription(
            Float32, 'sensor_readings', self.sensor_callback, 10)

        # Services
        self.health_check_srv = self.create_service(
            Trigger, 'system_health_check', self.health_check_callback)
        self.reset_srv = self.create_service(
            Trigger, 'system_reset', self.reset_callback)

        # Timers
        self.status_timer = self.create_timer(1.0, self.publish_status)
        self.sensor_timer = self.create_timer(0.5, self.publish_sensor_data)

        # State variables
        self.system_active = True
        self.sensor_value = 0.0
        self.message_count = 0

        self.get_logger().info('Communication Demo Node initialized')

    def status_callback(self, msg):
        """Handle incoming status messages"""
        self.get_logger().info(f'Received status: {msg.data}')
        self.message_count += 1

    def sensor_callback(self, msg):
        """Handle incoming sensor messages"""
        self.sensor_value = msg.data
        self.get_logger().info(f'Received sensor: {msg.data}')

    def publish_status(self):
        """Publish system status periodically"""
        if self.system_active:
            status_msg = String()
            status_msg.data = f'System active, messages processed: {self.message_count}, sensor: {self.sensor_value:.2f}'
            self.status_pub.publish(status_msg)

    def publish_sensor_data(self):
        """Publish simulated sensor data"""
        sensor_msg = Float32()
        sensor_msg.data = random.uniform(10, 50) + (self.message_count * 0.1)  # Slightly increasing with time
        self.sensor_pub.publish(sensor_msg)

    def health_check_callback(self, request, response):
        """Handle health check service request"""
        response.success = self.system_active
        if self.system_active:
            response.message = f'System healthy: {self.message_count} messages processed'
        else:
            response.message = 'System is inactive'

        self.get_logger().info(f'Health check: {response.message}')
        return response

    def reset_callback(self, request, response):
        """Handle system reset service request"""
        self.message_count = 0
        self.system_active = True
        response.success = True
        response.message = 'System reset successfully'

        self.get_logger().info('System reset executed')
        return response

def run_simulation():
    """Run a simulation that interacts with the demo node"""
    rclpy.init()

    # Create a separate node for simulation
    sim_node = rclpy.create_node('simulation_node')

    # Publishers
    cmd_pub = sim_node.create_publisher(Twist, 'navigation_commands', 10)

    # Service clients
    health_cli = sim_node.create_client(Trigger, 'system_health_check')
    reset_cli = sim_node.create_client(Trigger, 'system_reset')

    # Wait for services
    while not health_cli.wait_for_service(timeout_sec=1.0):
        sim_node.get_logger().info('Waiting for health service...')
    while not reset_cli.wait_for_service(timeout_sec=1.0):
        sim_node.get_logger().info('Waiting for reset service...')

    # Send some navigation commands
    for i in range(10):
        cmd = Twist()
        cmd.linear.x = random.uniform(0.1, 0.5)
        cmd.angular.z = random.uniform(-0.3, 0.3)
        cmd_pub.publish(cmd)

        # Call health check periodically
        if i % 3 == 0:
            req = Trigger.Request()
            future = health_cli.call_async(req)
            rclpy.spin_until_future_complete(sim_node, future)

            try:
                response = future.result()
                sim_node.get_logger().info(f'Simulation health: {response.message}')
            except Exception as e:
                sim_node.get_logger().error(f'Health check failed: {e}')

        time.sleep(1)

    sim_node.destroy_node()
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    demo_node = CommunicationDemoNode()

    # Run simulation in a separate thread
    sim_thread = threading.Thread(target=run_simulation)
    sim_thread.daemon = True
    sim_thread.start()

    try:
        rclpy.spin(demo_node)
    except KeyboardInterrupt:
        demo_node.get_logger().info('Shutting down...')
    finally:
        demo_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Run this comprehensive example:
```bash
python3 ~/ros2_ws/src/ros2_communication_demo.py
```

Expected results: The system demonstrates all communication patterns working together in a coordinated manner.

## Troubleshooting

### Common Error 1: Topic Connection Issues
**Cause**: Nodes not connecting to topics due to network or configuration issues
**Solution**: Check ROS_DOMAIN_ID, network settings, and topic names
**Prevention Tips**: Use consistent naming conventions and environment setup

### Common Error 2: Service Timeout
**Cause**: Service client timing out waiting for server response
**Solution**: Ensure service server is running and check network connectivity
**Prevention Tips**: Implement proper service availability checking

### Common Error 3: Memory Issues with Data History
**Cause**: Storing too much historical data without limits
**Solution**: Implement data pruning and limits
**Prevention Tips**: Always implement data lifecycle management

## Key Takeaways

- Nodes provide modular execution units for specific functions
- Topics enable asynchronous, decoupled communication
- Services provide synchronous request-response patterns
- Quality of Service settings allow fine-tuning of communication behavior
- Proper error handling and state management are essential for robust systems

## Additional Resources

- [ROS 2 Nodes Documentation](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Single-Package-Define-And-Use-Interface.html)
- [ROS 2 Topics and Services](https://docs.ros.org/en/humble/Tutorials/Topics/Understanding-ROS2-Topics.html)
- [Quality of Service in ROS 2](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html)
- [ROS 2 Services Tutorial](https://docs.ros.org/en/humble/Tutorials/Services/Understanding-ROS2-Services.html)

## Self-Assessment

1. What is the difference between a publisher-subscriber pattern and a client-service pattern?
2. When would you use QoS reliability settings in ROS 2?
3. How do you handle multiple subscriptions in a single node?
4. What are the advantages of using services over topics for certain operations?
5. How would you design a ROS 2 system with multiple publishers and subscribers?

<ChapterNavigation
  previous={{
    permalink: '/docs/module-01-ros2/chapter-01-01-architecture',
    title: '1.1 ROS 2 Architecture & Core Concepts'
  }}
  next={{
    permalink: '/docs/module-01-ros2/chapter-01-03-workspaces-packages',
    title: '1.3 Workspaces and Packages'
  }}
/>