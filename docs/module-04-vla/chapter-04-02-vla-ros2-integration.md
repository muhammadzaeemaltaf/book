---
id: chapter-04-02-vla-ros2-integration
title: "Chapter 4.2: VLA-ROS2 Integration"
sidebar_label: "4.2 VLA-ROS2 Integration"
sidebar_position: 2
description: "Integrate Vision-Language-Action models with ROS 2: multimodal action servers, distributed inference, and real-time communication patterns for humanoid robotics."
keywords: [VLA, ROS 2, multimodal AI, action servers, distributed inference, humanoid control, message passing]
---

# Chapter 4.2: VLA-ROS2 Integration

## Learning Objectives

By the end of this chapter, you will be able to:

- **Design** multimodal ROS 2 action servers for VLA execution
- **Implement** distributed VLA inference across robot systems
- **Create** custom message types for multimodal communication
- **Integrate** VLA models with existing ROS 2 navigation and manipulation stacks
- **Optimize** real-time performance for VLA-based robot control
- **Debug** multimodal systems using ROS 2 tools

## Prerequisites

- Understanding of VLA fundamentals (Chapter 4.1)
- Knowledge of ROS 2 action servers and clients
- Experience with custom message and service definitions
- Familiarity with ROS 2 launch files and parameter management
- Basic understanding of VLA model architectures

---

## 4.2.1 Multimodal Message Types for VLA

### Custom Message Definitions

**msg/VLACommand.msg:**
```
# Vision-Language-Action command message
string command_text
sensor_msgs/Image[] camera_images
geometry_msgs/PoseStamped[] target_poses
float64[] confidence_thresholds
bool use_safety_monitoring
string[] required_sensors
```

**msg/VLAAction.msg:**
```
# VLA action output
geometry_msgs/TwistStamped[] motion_commands
sensor_msgs/JointState[] joint_commands
std_msgs/Float32[] gripper_commands
geometry_msgs/PoseStamped[] end_effector_poses
float64[] confidence_scores
string[] action_descriptions
```

**msg/VLAResult.msg:**
```
# VLA execution result
bool success
string error_message
float64 execution_time
sensor_msgs/Image[] result_images
geometry_msgs/PoseStamped[] final_poses
float64[] confidence_scores
builtin_interfaces/Time timestamp
```

### Creating Custom Messages

```bash
# Create message directory
mkdir -p ~/ros2_ws/src/vla_interfaces/msg

# Create message files
cat > ~/ros2_ws/src/vla_interfaces/msg/VLACommand.msg << 'EOF'
# Vision-Language-Action command message
string command_text
sensor_msgs/Image[] camera_images
geometry_msgs/PoseStamped[] target_poses
float64[] confidence_thresholds
bool use_safety_monitoring
string[] required_sensors
EOF

cat > ~/ros2_ws/src/vla_interfaces/msg/VLAAction.msg << 'EOF'
# VLA action output
geometry_msgs/TwistStamped[] motion_commands
sensor_msgs/JointState[] joint_commands
std_msgs/Float32[] gripper_commands
geometry_msgs/PoseStamped[] end_effector_poses
float64[] confidence_scores
string[] action_descriptions
EOF

cat > ~/ros2_ws/src/vla_interfaces/msg/VLAResult.msg << 'EOF'
# VLA execution result
bool success
string error_message
float64 execution_time
sensor_msgs/Image[] result_images
geometry_msgs/PoseStamped[] final_poses
float64[] confidence_scores
builtin_interfaces/Time timestamp
EOF
```

### Package.xml for Custom Messages

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>vla_interfaces</name>
  <version>0.0.0</version>
  <description>Custom message types for Vision-Language-Action systems</description>
  <maintainer email="robotics@example.com">Robotics Team</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>rosidl_default_generators</buildtool_depend>

  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>builtin_interfaces</depend>

  <exec_depend>rosidl_default_runtime</exec_depend>

  <member_of_group>rosidl_interface_packages</member_of_group>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### CMakeLists.txt for Message Generation

```cmake
cmake_minimum_required(VERSION 3.8)
project(vla_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Generate messages
set(msg_files
  "msg/VLACommand.msg"
  "msg/VLAAction.msg"
  "msg/VLAResult.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs builtin_interfaces
)

ament_package()
```

---

## 4.2.2 VLA Action Server Implementation

### Action Definition

**action/VLAManipulation.action:**
```
# Goal: Execute a VLA-based manipulation task
string command_text
sensor_msgs/Image initial_scene_image
geometry_msgs/PoseStamped target_object_pose
float64 confidence_threshold
---
# Result: Outcome of the manipulation
bool success
string error_message
geometry_msgs/PoseStamped final_object_pose
geometry_msgs/PoseStamped final_end_effector_pose
float64 execution_time
float64 confidence_score
---
# Feedback: Real-time progress updates
string current_action
float64 progress_percentage
sensor_msgs/Image current_scene
geometry_msgs/PoseStamped current_end_effector_pose
float64 current_confidence
```

### Action Server Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import torch
import numpy as np

from vla_interfaces.action import VLAManipulation
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, TwistStamped
from cv_bridge import CvBridge

class VLAActionServer(Node):
    """
    Action server for Vision-Language-Action manipulation tasks.
    """

    def __init__(self):
        super().__init__('vla_action_server')

        # Initialize VLA model
        self.vla_model = self.load_vla_model()
        self.vla_model.eval()

        # ROS interfaces
        self.bridge = CvBridge()
        self.action_server = ActionServer(
            self,
            VLAManipulation,
            'vla_manipulation',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publishers for monitoring
        self.action_pub = self.create_publisher(JointState, '/vla_predicted_actions', 10)
        self.confidence_pub = self.create_publisher(TwistStamped, '/vla_confidence', 10)

        # State management
        self._active_goal = None
        self._cancel_requested = False
        self._execution_lock = threading.Lock()

        self.get_logger().info("VLA Action Server initialized")

    def load_vla_model(self):
        """
        Load pre-trained VLA model.
        In practice, this would load from checkpoint.
        """
        # Placeholder - replace with actual VLA model
        from vla_humanoid_control.models.simple_vla import SimpleVLA
        model = SimpleVLA()
        # model.load_state_dict(torch.load('path/to/vla_model.pth'))
        return model

    def goal_callback(self, goal_request):
        """
        Accept or reject goal requests.
        """
        self.get_logger().info(f"Received VLA manipulation goal: {goal_request.command_text}")

        # Check if we can accept the goal
        if self._active_goal is not None:
            return GoalResponse.REJECT
        else:
            return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Accept or reject cancel requests.
        """
        self.get_logger().info("Received cancel request")
        self._cancel_requested = True
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """
        Execute the VLA manipulation task.
        """
        self.get_logger().info("Executing VLA manipulation goal")

        # Store goal handle
        with self._execution_lock:
            self._active_goal = goal_handle
            self._cancel_requested = False

        # Extract goal parameters
        command_text = goal_handle.request.command_text
        initial_image = goal_handle.request.initial_scene_image
        target_pose = goal_handle.request.target_object_pose
        confidence_threshold = goal_handle.request.confidence_threshold

        # Initialize feedback
        feedback_msg = VLAManipulation.Feedback()
        result = VLAManipulation.Result()

        try:
            # Process initial image
            cv_image = self.bridge.imgmsg_to_cv2(initial_image, 'rgb8')
            image_tensor = torch.from_numpy(cv_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Tokenize command
            tokenizer = SimpleTokenizer()  # From Chapter 4.1
            command_tokens = tokenizer.tokenize(command_text).unsqueeze(0)

            # Initialize progress tracking
            total_steps = 100  # Simulated execution steps
            current_step = 0

            # Execute manipulation sequence
            while current_step < total_steps and not self._cancel_requested:
                # Get VLA prediction
                with torch.no_grad():
                    vla_output = self.vla_model(image_tensor, command_tokens)

                actions = vla_output['actions'].squeeze(0).numpy()
                confidence = vla_output['confidence'].squeeze(0).item()

                # Check confidence
                if confidence < confidence_threshold:
                    result.success = False
                    result.error_message = f"Low confidence: {confidence:.2f} < {confidence_threshold}"
                    goal_handle.abort()
                    return result

                # Publish confidence for monitoring
                confidence_msg = TwistStamped()
                confidence_msg.header.stamp = self.get_clock().now().to_msg()
                confidence_msg.twist.linear.x = confidence
                self.confidence_pub.publish(confidence_msg)

                # Update feedback
                feedback_msg.current_action = f"Executing step {current_step + 1}"
                feedback_msg.progress_percentage = float(current_step) / total_steps * 100.0
                feedback_msg.current_scene = initial_image
                feedback_msg.current_confidence = confidence

                # Publish feedback
                goal_handle.publish_feedback(feedback_msg)

                # Execute action (simulated)
                self.execute_action(actions)

                # Update image for next iteration (simulated)
                # In practice: get new image from robot's camera
                current_step += 1

                # Sleep to simulate execution time
                time.sleep(0.05)

            # Check if canceled
            if self._cancel_requested:
                result.success = False
                result.error_message = "Goal canceled"
                goal_handle.canceled()
            else:
                # Success case
                result.success = True
                result.error_message = ""
                result.execution_time = current_step * 0.05
                result.confidence_score = confidence

                # Set final poses (simulated)
                result.final_object_pose = target_pose
                result.final_end_effector_pose = PoseStamped()
                result.final_end_effector_pose.pose.position.x = 0.5
                result.final_end_effector_pose.pose.position.y = 0.0
                result.final_end_effector_pose.pose.position.z = 0.8

                goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f"Error in VLA execution: {str(e)}")
            result.success = False
            result.error_message = f"Execution error: {str(e)}"
            goal_handle.abort()

        finally:
            # Clean up
            with self._execution_lock:
                self._active_goal = None
                self._cancel_requested = False

        return result

    def execute_action(self, actions):
        """
        Execute the predicted actions on the robot.
        """
        # Convert actions to robot commands
        joint_commands = JointState()
        joint_commands.header.stamp = self.get_clock().now().to_msg()
        joint_commands.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        joint_commands.position = actions.tolist()

        # Publish commands
        self.action_pub.publish(joint_commands)

    def destroy(self):
        """
        Clean up action server.
        """
        self.action_server.destroy()
        super().destroy_node()

class SimpleTokenizer:
    """
    Simple tokenizer for demonstration (same as Chapter 4.1).
    """
    def __init__(self):
        self.vocab = {
            'go': 1, 'move': 2, 'forward': 3, 'backward': 4,
            'left': 5, 'right': 6, 'turn': 7, 'stop': 8,
            'pick': 9, 'place': 10, 'grasp': 11, 'release': 12,
            'the': 13, 'a': 14, 'an': 15, 'and': 16, 'or': 17,
            'up': 18, 'down': 19, 'to': 20, 'at': 21, 'on': 22,
            'red': 23, 'blue': 24, 'green': 25, 'cup': 26, 'box': 27,
            'table': 28, 'chair': 29, 'kitchen': 30, 'room': 31
        }
        self.unk_token = 0

    def tokenize(self, text):
        words = text.lower().split()
        token_ids = [self.vocab.get(word, self.unk_token) for word in words]

        if len(token_ids) < 10:
            token_ids.extend([0] * (10 - len(token_ids)))
        else:
            token_ids = token_ids[:10]

        return torch.tensor(token_ids, dtype=torch.long)

def main(args=None):
    rclpy.init(args=args)

    # Use multi-threaded executor for action server
    executor = MultiThreadedExecutor()

    action_server = VLAActionServer()
    executor.add_node(action_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        action_server.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Client Example

```python
#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from vla_interfaces.action import VLAManipulation

class VLAActionClient(Node):
    """
    Action client for sending VLA manipulation requests.
    """

    def __init__(self):
        super().__init__('vla_action_client')
        self._action_client = ActionClient(self, VLAManipulation, 'vla_manipulation')
        self.bridge = CvBridge()

    def send_goal(self, command_text, target_pose=None):
        """
        Send a VLA manipulation goal.
        """
        goal_msg = VLAManipulation.Goal()
        goal_msg.command_text = command_text
        goal_msg.confidence_threshold = 0.7

        # Get current scene image
        # In practice: subscribe to camera topic or get from service
        goal_msg.initial_scene_image = self.get_current_scene_image()

        if target_pose is not None:
            goal_msg.target_object_pose = target_pose

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handle goal response.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """
        Handle result callback.
        """
        result = future.result().result
        self.get_logger().info(f'Result: {result.success}, Error: {result.error_message}')

    def feedback_callback(self, feedback_msg):
        """
        Handle feedback callback.
        """
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback: {feedback.current_action}, '
            f'Progress: {feedback.progress_percentage:.1f}%, '
            f'Confidence: {feedback.current_confidence:.2f}'
        )

    def get_current_scene_image(self):
        """
        Get current scene image (simulated).
        In practice: get from camera topic.
        """
        # Create a dummy image for demonstration
        import numpy as np
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return self.bridge.cv2_to_imgmsg(dummy_image, encoding='bgr8')

def main(args=None):
    rclpy.init(args=args)

    action_client = VLAActionClient()

    # Send a test goal
    target_pose = PoseStamped()
    target_pose.pose.position.x = 0.5
    target_pose.pose.position.y = 0.0
    target_pose.pose.position.z = 0.8

    action_client.send_goal("pick up the red cup", target_pose)

    # Spin to process callbacks
    rclpy.spin(action_client)

    action_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 4.2.3 Distributed VLA Inference Architecture

### Multi-Node VLA System

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Perception    │    │   VLA Inference  │    │  Robot Control  │
│   Node          │───→│   Node           │───→│   Node          │
│                 │    │                  │    │                 │
│ - Camera        │    │ - VLA Model      │    │ - Joint Control │
│ - IMU           │    │ - Language Proc. │    │ - Trajectory    │
│ - LiDAR         │    │ - Action Gen.    │    │ - Safety Mon.   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               │
                       ┌─────────────────┐
                       │  Central Node   │
                       │                 │
                       │ - Coordination  │
                       │ - Monitoring    │
                       │ - Logging       │
                       └─────────────────┘
```

### VLA Inference Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import threading
import queue
import time

from vla_interfaces.msg import VLACommand, VLAAction

class VLAInferenceNode(Node):
    """
    Distributed VLA inference node.
    Handles model inference separately from perception and control.
    """

    def __init__(self):
        super().__init__('vla_inference_node')

        # Initialize VLA model
        self.vla_model = self.load_vla_model()
        self.vla_model.eval()

        # Thread-safe queues for processing
        self.inference_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)

        # ROS interfaces
        self.bridge = CvBridge()

        # Subscribers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.command_sub = self.create_subscription(
            VLACommand,
            'vla_command',
            self.command_callback,
            qos_profile
        )

        self.image_sub = self.create_subscription(
            Image,
            'camera/rgb/image_raw',
            self.image_callback,
            qos_profile
        )

        # Publishers
        self.action_pub = self.create_publisher(VLAAction, 'vla_predicted_actions', 10)
        self.confidence_pub = self.create_publisher(Float32, 'vla_confidence', 10)

        # Processing thread
        self.processing_thread = threading.Thread(target=self.process_inference_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Current state
        self.current_image = None
        self.current_command = None
        self.image_lock = threading.Lock()
        self.command_lock = threading.Lock()

        self.get_logger().info("VLA Inference Node initialized")

    def load_vla_model(self):
        """
        Load VLA model with optimized inference settings.
        """
        from vla_humanoid_control.models.simple_vla import SimpleVLA
        model = SimpleVLA()

        # Optimize for inference
        if torch.cuda.is_available():
            model = model.cuda()
            self.get_logger().info("Using GPU for VLA inference")
        else:
            self.get_logger().info("Using CPU for VLA inference")

        return model

    def command_callback(self, msg):
        """
        Handle incoming VLA commands.
        """
        with self.command_lock:
            self.current_command = msg

    def image_callback(self, msg):
        """
        Handle incoming images for inference.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

            # Resize for model input
            import cv2
            resized_image = cv2.resize(cv_image, (224, 224))

            with self.image_lock:
                self.current_image = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0

            # Add to inference queue if we have both image and command
            with self.image_lock, self.command_lock:
                if self.current_image is not None and self.current_command is not None:
                    try:
                        self.inference_queue.put_nowait({
                            'image': self.current_image.clone(),
                            'command': self.current_command.command_text,
                            'timestamp': msg.header.stamp
                        })
                    except queue.Full:
                        self.get_logger().warn("Inference queue full, dropping frame")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def process_inference_loop(self):
        """
        Background thread for VLA inference processing.
        """
        while rclpy.ok():
            try:
                # Get item from queue
                item = self.inference_queue.get(timeout=1.0)

                # Perform inference
                result = self.perform_inference(item)

                # Publish result
                self.publish_result(result)

                self.inference_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error in inference loop: {e}")

    def perform_inference(self, item):
        """
        Perform VLA inference on single item.
        """
        image = item['image']
        command = item['command']
        timestamp = item['timestamp']

        # Prepare inputs
        image_batch = image.unsqueeze(0)  # Add batch dimension

        # Tokenize command
        tokenizer = SimpleTokenizer()
        command_tokens = tokenizer.tokenize(command).unsqueeze(0)

        # Perform inference
        start_time = time.time()
        with torch.no_grad():
            if torch.cuda.is_available():
                image_batch = image_batch.cuda()
                command_tokens = command_tokens.cuda()

            output = self.vla_model(image_batch, command_tokens)

            # Move results back to CPU for ROS publishing
            actions = output['actions'].squeeze(0).cpu().numpy()
            confidence = output['confidence'].squeeze(0).cpu().item()

        inference_time = time.time() - start_time

        return {
            'actions': actions,
            'confidence': confidence,
            'timestamp': timestamp,
            'inference_time': inference_time,
            'command': command
        }

    def publish_result(self, result):
        """
        Publish VLA inference results.
        """
        # Create VLAAction message
        action_msg = VLAAction()
        action_msg.header.stamp = result['timestamp']
        action_msg.header.frame_id = 'base_link'

        # Convert actions to appropriate message types
        # For this example, we'll create a simple joint command
        joint_cmd = JointState()
        joint_cmd.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        joint_cmd.position = result['actions'].tolist()
        joint_cmd.header.stamp = result['timestamp']

        action_msg.joint_commands = [joint_cmd]
        action_msg.confidence_scores = [result['confidence']]
        action_msg.action_descriptions = [result['command']]

        # Publish action
        self.action_pub.publish(action_msg)

        # Publish confidence
        confidence_msg = Float32()
        confidence_msg.data = result['confidence']
        self.confidence_pub.publish(confidence_msg)

        self.get_logger().info(
            f"VLA inference completed: confidence={result['confidence']:.2f}, "
            f"time={result['inference_time']:.3f}s"
        )

    def destroy_node(self):
        """
        Clean up before node destruction.
        """
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VLAInferenceNode()

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

### VLA Control Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool

from vla_interfaces.msg import VLAAction
import threading
import time

class VLAControlNode(Node):
    """
    VLA control node that executes VLA-predicted actions.
    """

    def __init__(self):
        super().__init__('vla_control_node')

        # ROS interfaces
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.vla_action_sub = self.create_subscription(
            VLAAction,
            'vla_predicted_actions',
            self.vla_action_callback,
            qos_profile
        )

        self.confidence_sub = self.create_subscription(
            Float32,
            'vla_confidence',
            self.confidence_callback,
            qos_profile
        )

        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.safety_pub = self.create_publisher(Bool, '/safety_status', 10)

        # Parameters
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('use_safety_monitoring', True)

        # State
        self.current_confidence = 0.0
        self.safety_enabled = True
        self.confidence_lock = threading.Lock()

        # Safety timer
        self.safety_timer = self.create_timer(0.1, self.safety_check)

        self.get_logger().info("VLA Control Node initialized")

    def vla_action_callback(self, msg):
        """
        Handle VLA action commands.
        """
        if not self.safety_enabled:
            self.get_logger().warn("Safety disabled, not executing VLA action")
            return

        with self.confidence_lock:
            if self.current_confidence < self.get_parameter('confidence_threshold').value:
                self.get_logger().warn(
                    f"Low confidence ({self.current_confidence:.2f}), not executing action"
                )
                return

        # Execute joint commands
        if msg.joint_commands:
            for joint_cmd in msg.joint_commands:
                # Apply velocity limits
                max_vel = self.get_parameter('max_velocity').value
                limited_positions = [
                    max(-max_vel, min(max_vel, pos)) for pos in joint_cmd.position
                ]
                joint_cmd.position = limited_positions

                # Add timestamp
                joint_cmd.header.stamp = self.get_clock().now().to_msg()

                # Publish command
                self.joint_cmd_pub.publish(joint_cmd)

                self.get_logger().info(f"Executed VLA joint command: {joint_cmd.position}")

    def confidence_callback(self, msg):
        """
        Update current confidence level.
        """
        with self.confidence_lock:
            self.current_confidence = msg.data

    def safety_check(self):
        """
        Periodic safety check.
        """
        safety_status = Bool()
        safety_status.data = self.safety_enabled

        # Additional safety checks could go here
        # - Joint limits
        # - Collision detection
        # - Balance monitoring
        # - Emergency stop conditions

        self.safety_pub.publish(safety_status)

    def enable_safety(self, enable=True):
        """
        Enable or disable safety monitoring.
        """
        self.safety_enabled = enable
        self.get_logger().info(f"VLA safety {'enabled' if enable else 'disabled'}")

def main(args=None):
    rclpy.init(args=args)
    node = VLAControlNode()

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

---

## 4.2.4 VLA-ROS2 Navigation Integration

### VLA Navigation Action Server

```python
#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from vla_interfaces.action import VLAManipulation  # Reusing for navigation

class VLANavigationServer(Node):
    """
    VLA-based navigation action server.
    Integrates with Nav2 for language-guided navigation.
    """

    def __init__(self):
        super().__init__('vla_navigation_server')

        # Initialize VLA model for navigation
        self.vla_model = self.load_navigation_vla_model()
        self.vla_model.eval()

        # ROS interfaces
        self.bridge = CvBridge()

        # Action server for navigation tasks
        self.nav_action_server = ActionServer(
            self,
            VLAManipulation,  # Reusing action type for simplicity
            'vla_navigation',
            execute_callback=self.navigation_execute_callback
        )

        # Subscribers for navigation
        self.image_sub = self.create_subscription(
            Image,
            'camera/rgb/image_raw',
            self.image_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
        )

        # Publisher for navigation goals
        self.nav_goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)

        # Current state
        self.current_image = None
        self.image_lock = threading.Lock()

        self.get_logger().info("VLA Navigation Server initialized")

    def load_navigation_vla_model(self):
        """
        Load VLA model specialized for navigation tasks.
        """
        # For navigation, we might use a different model architecture
        # or fine-tune the general VLA model for navigation
        from vla_humanoid_control.models.simple_vla import SimpleVLA
        model = SimpleVLA()
        return model

    def image_callback(self, msg):
        """
        Process incoming navigation images.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            import cv2
            resized_image = cv2.resize(cv_image, (224, 224))

            with self.image_lock:
                self.current_image = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            self.get_logger().error(f"Error processing navigation image: {e}")

    def navigation_execute_callback(self, goal_handle):
        """
        Execute VLA-based navigation task.
        """
        self.get_logger().info(f"Executing VLA navigation: {goal_handle.request.command_text}")

        command_text = goal_handle.request.command_text

        # Process current image and command
        with self.image_lock:
            if self.current_image is None:
                result = VLAManipulation.Result()
                result.success = False
                result.error_message = "No current image available"
                goal_handle.abort()
                return result

            current_image = self.current_image.clone()

        # Perform VLA inference for navigation
        try:
            image_tensor = current_image.unsqueeze(0)
            tokenizer = SimpleTokenizer()
            command_tokens = tokenizer.tokenize(command_text).unsqueeze(0)

            with torch.no_grad():
                vla_output = self.vla_model(image_tensor, command_tokens)

            actions = vla_output['actions'].squeeze(0).numpy()
            confidence = vla_output['confidence'].squeeze(0).item()

            # Convert VLA output to navigation goal
            nav_goal = self.convert_vla_to_navigation_goal(actions, command_text)

            if nav_goal is not None:
                # Publish navigation goal
                self.nav_goal_pub.publish(nav_goal)

                # Wait for navigation to complete (simplified)
                # In practice: monitor Nav2 status and provide feedback
                import time
                time.sleep(5.0)  # Simulated navigation time

                # Success result
                result = VLAManipulation.Result()
                result.success = True
                result.confidence_score = confidence
                result.error_message = ""

                goal_handle.succeed()
                return result
            else:
                result = VLAManipulation.Result()
                result.success = False
                result.error_message = "Could not generate navigation goal from VLA output"
                goal_handle.abort()
                return result

        except Exception as e:
            self.get_logger().error(f"Error in VLA navigation: {e}")
            result = VLAManipulation.Result()
            result.success = False
            result.error_message = f"Navigation error: {str(e)}"
            goal_handle.abort()
            return result

    def convert_vla_to_navigation_goal(self, actions, command_text):
        """
        Convert VLA actions to navigation goal pose.
        This is a simplified implementation - in practice, this would
        involve more sophisticated parsing and mapping.
        """
        # Parse command to determine navigation intent
        command_lower = command_text.lower()

        # Simple mapping based on command keywords
        if 'kitchen' in command_lower:
            target_x, target_y = 2.0, 1.0
        elif 'living room' in command_lower:
            target_x, target_y = -1.0, 0.5
        elif 'bedroom' in command_lower:
            target_x, target_y = 0.0, -2.0
        else:
            # Use VLA actions to determine target
            target_x = float(actions[0]) * 2.0  # Scale from [-1,1] to [-2,2]
            target_y = float(actions[1]) * 2.0

        # Create navigation goal
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = target_x
        goal.pose.position.y = target_y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0  # No rotation for simplicity

        return goal

def main(args=None):
    rclpy.init(args=args)
    node = VLANavigationServer()

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

---

## 4.2.5 Performance Optimization for VLA-ROS2

### GPU-Accelerated Inference

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import torch
import torch_tensorrt

class OptimizedVLAInferenceNode(Node):
    """
    GPU-optimized VLA inference node using TensorRT.
    """

    def __init__(self):
        super().__init__('optimized_vla_inference')

        # Initialize optimized model
        self.optimized_model = self.load_optimized_model()

        # ROS interfaces
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            'camera/rgb/image_raw',
            self.image_callback,
            10
        )
        self.result_pub = self.create_publisher(Float32, 'vla_inference_time', 10)

        self.get_logger().info("Optimized VLA Inference Node initialized")

    def load_optimized_model(self):
        """
        Load and optimize VLA model using TensorRT.
        """
        # Load original model
        from vla_humanoid_control.models.simple_vla import SimpleVLA
        model = SimpleVLA()
        # model.load_state_dict(torch.load('vla_model.pth'))
        model.eval()

        # Optimize with TensorRT (if available)
        if torch_tensorrt is not None and torch.cuda.is_available():
            try:
                # Define optimization settings
                optimized_model = torch_tensorrt.compile(
                    model,
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=[1, 3, 224, 224],
                            opt_shape=[4, 3, 224, 224],
                            max_shape=[8, 3, 224, 224],
                            dtype=torch.float
                        ),
                        torch_tensorrt.Input(
                            min_shape=[1, 10],
                            opt_shape=[4, 10],
                            max_shape=[8, 10],
                            dtype=torch.long
                        )
                    ],
                    enabled_precisions={torch.float, torch.half},  # FP32 and FP16
                    workspace_size=1 << 28,  # 256MB
                    truncate_long_and_double=True
                )
                self.get_logger().info("Model optimized with TensorRT")
                return optimized_model
            except Exception as e:
                self.get_logger().warn(f"TensorRT optimization failed: {e}")

        # Fallback to regular model
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def image_callback(self, msg):
        """
        Process image with optimized inference.
        """
        try:
            import time
            start_time = time.time()

            # Process image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            import cv2
            resized_image = cv2.resize(cv_image, (224, 224))
            image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

            # Move to GPU if available
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()

            # Tokenize a dummy command for this example
            tokenizer = SimpleTokenizer()
            command_tokens = tokenizer.tokenize("move forward").unsqueeze(0)
            if torch.cuda.is_available():
                command_tokens = command_tokens.cuda()

            # Optimized inference
            with torch.no_grad():
                output = self.optimized_model(image_tensor, command_tokens)

            inference_time = time.time() - start_time

            # Publish timing info
            timing_msg = Float32()
            timing_msg.data = inference_time
            self.result_pub.publish(timing_msg)

            self.get_logger().info(f"Optimized inference time: {inference_time:.3f}s")

        except Exception as e:
            self.get_logger().error(f"Error in optimized inference: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = OptimizedVLAInferenceNode()

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

### Batch Processing for Efficiency

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import queue
import threading
import time

class BatchedVLAInferenceNode(Node):
    """
    Batch processing VLA inference for improved efficiency.
    """

    def __init__(self):
        super().__init__('batched_vla_inference')

        # Initialize model
        self.vla_model = self.load_vla_model()
        self.vla_model.eval()

        # Batch processing parameters
        self.batch_size = 4
        self.batch_timeout = 0.05  # 50ms timeout for batch accumulation

        # Processing queues
        self.input_queue = queue.Queue(maxsize=50)
        self.output_queue = queue.Queue(maxsize=50)

        # ROS interfaces
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, 'camera/rgb/image_raw', self.image_callback, 10)
        self.command_sub = self.create_subscription(String, 'vla_command', self.command_callback, 10)

        # Processing thread
        self.processing_thread = threading.Thread(target=self.batch_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info("Batched VLA Inference Node initialized")

    def load_vla_model(self):
        """
        Load VLA model for batch processing.
        """
        from vla_humanoid_control.models.simple_vla import SimpleVLA
        model = SimpleVLA()
        return model

    def image_callback(self, msg):
        """
        Add image to processing queue.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            import cv2
            resized_image = cv2.resize(cv_image, (224, 224))
            image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0

            item = {
                'image': image_tensor,
                'timestamp': msg.header.stamp,
                'type': 'image'
            }

            try:
                self.input_queue.put_nowait(item)
            except queue.Full:
                self.get_logger().warn("Input queue full, dropping image")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def command_callback(self, msg):
        """
        Add command to processing queue.
        """
        item = {
            'command': msg.data,
            'timestamp': self.get_clock().now().to_msg(),
            'type': 'command'
        }

        try:
            self.input_queue.put_nowait(item)
        except queue.Full:
            self.get_logger().warn("Input queue full, dropping command")

    def batch_processing_loop(self):
        """
        Batch processing loop for efficient inference.
        """
        batch = []
        last_process_time = time.time()

        while rclpy.ok():
            try:
                # Get item from queue with timeout
                item = self.input_queue.get(timeout=0.1)

                # Add to batch based on type
                if item['type'] == 'image':
                    # Look for corresponding command
                    # In practice: match images with commands based on timestamps
                    batch.append(item)
                elif item['type'] == 'command':
                    # Store command for matching
                    self.current_command = item['command']

                # Process batch if full or timeout
                current_time = time.time()
                if (len(batch) >= self.batch_size or
                    (batch and current_time - last_process_time > self.batch_timeout)):

                    if batch and hasattr(self, 'current_command'):
                        self.process_batch(batch)
                        batch = []
                        last_process_time = current_time

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error in batch processing: {e}")

    def process_batch(self, batch):
        """
        Process a batch of images.
        """
        try:
            # Prepare batch tensors
            images = torch.stack([item['image'] for item in batch])
            images = images.unsqueeze(0).expand(self.batch_size, -1, -1, -1)  # Expand to batch size

            # Prepare command tokens (same for all in batch for this example)
            tokenizer = SimpleTokenizer()
            command_tokens = tokenizer.tokenize(self.current_command)
            command_tokens = command_tokens.unsqueeze(0).expand(self.batch_size, -1)  # Expand to batch size

            # Batch inference
            start_time = time.time()
            with torch.no_grad():
                output = self.vla_model(images, command_tokens)
            inference_time = time.time() - start_time

            self.get_logger().info(
                f"Processed batch of {len(batch)} items in {inference_time:.3f}s "
                f"(avg: {inference_time/len(batch):.3f}s per item)"
            )

        except Exception as e:
            self.get_logger().error(f"Error processing batch: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BatchedVLAInferenceNode()

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

---

## 4.2.6 VLA-ROS2 System Launch Files

### Complete VLA System Launch

**launch/vla_system.launch.py:**

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    confidence_threshold = LaunchConfiguration('confidence_threshold', default='0.6')

    return LaunchDescription([
        # VLA Perception Node
        Node(
            package='vla_humanoid_control',
            executable='vla_perception_node',
            name='vla_perception',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/camera/rgb/image_raw', '/humanoid/camera/rgb/image_raw'),
                ('/imu/data', '/humanoid/imu/data')
            ]
        ),

        # VLA Inference Node
        Node(
            package='vla_humanoid_control',
            executable='vla_inference_node',
            name='vla_inference',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/vla_command', '/vla_command'),
                ('/camera/rgb/image_raw', '/humanoid/camera/rgb/image_raw'),
                ('/vla_predicted_actions', '/vla_predicted_actions')
            ]
        ),

        # VLA Control Node
        Node(
            package='vla_humanoid_control',
            executable='vla_control_node',
            name='vla_control',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'confidence_threshold': confidence_threshold}
            ],
            remappings=[
                ('/vla_predicted_actions', '/vla_predicted_actions'),
                ('/joint_commands', '/humanoid/joint_commands'),
                ('/vla_confidence', '/vla_confidence')
            ]
        ),

        # VLA Navigation Server
        Node(
            package='vla_humanoid_control',
            executable='vla_navigation_server',
            name='vla_navigation',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/camera/rgb/image_raw', '/humanoid/camera/rgb/image_raw'),
                ('/goal_pose', '/navigate_to_pose')
            ]
        ),

        # VLA Action Server
        Node(
            package='vla_humanoid_control',
            executable='vla_action_server',
            name='vla_action_server',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        )
    ])
```

### Parameter Configuration

**config/vla_params.yaml:**

```yaml
vla_perception_node:
  ros__parameters:
    image_topic: "/humanoid/camera/rgb/image_raw"
    depth_topic: "/humanoid/camera/depth/image_raw"
    imu_topic: "/humanoid/imu/data"
    image_processing_rate: 10.0
    use_compressed_images: false

vla_inference_node:
  ros__parameters:
    confidence_threshold: 0.7
    inference_rate: 10.0
    batch_size: 1
    use_gpu: true
    model_path: "/path/to/vla_model.pth"
    max_queue_size: 10

vla_control_node:
  ros__parameters:
    confidence_threshold: 0.6
    max_velocity: 0.5
    use_safety_monitoring: true
    joint_command_topic: "/humanoid/joint_commands"
    safety_check_rate: 100.0

vla_navigation_server:
  ros__parameters:
    navigation_frame: "map"
    goal_tolerance: 0.2
    use_vla_for_navigation: true
    navigation_confidence_threshold: 0.5
```

---

## 4.2.7 VLA System Monitoring and Debugging

### VLA System Monitor Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Bool, String
from sensor_msgs.msg import Image
import time

class VLAMonitorNode(Node):
    """
    Monitor VLA system performance and health.
    """

    def __init__(self):
        super().__init__('vla_monitor')

        # QoS profile for monitoring
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers for monitoring
        self.confidence_sub = self.create_subscription(
            Float32, 'vla_confidence', self.confidence_callback, qos_profile)
        self.inference_time_sub = self.create_subscription(
            Float32, 'vla_inference_time', self.inference_time_callback, qos_profile)
        self.safety_status_sub = self.create_subscription(
            Bool, 'safety_status', self.safety_status_callback, qos_profile)

        # Publishers for alerts
        self.alert_pub = self.create_publisher(String, 'vla_alerts', 10)

        # Performance tracking
        self.confidence_history = []
        self.inference_time_history = []
        self.safety_status = True

        # Monitoring timer
        self.monitor_timer = self.create_timer(1.0, self.monitor_callback)

        self.get_logger().info("VLA Monitor Node initialized")

    def confidence_callback(self, msg):
        """
        Track confidence values.
        """
        self.confidence_history.append(msg.data)
        if len(self.confidence_history) > 100:  # Keep last 100 values
            self.confidence_history.pop(0)

    def inference_time_callback(self, msg):
        """
        Track inference performance.
        """
        self.inference_time_history.append(msg.data)
        if len(self.inference_time_history) > 100:
            self.inference_time_history.pop(0)

    def safety_status_callback(self, msg):
        """
        Track safety status.
        """
        self.safety_status = msg.data

    def monitor_callback(self):
        """
        Periodic monitoring and alert generation.
        """
        # Check confidence levels
        if self.confidence_history:
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
            if avg_confidence < 0.5:
                self.send_alert(f"LOW_CONFIDENCE: Average confidence {avg_confidence:.2f} < 0.5")

        # Check inference performance
        if self.inference_time_history:
            avg_time = sum(self.inference_time_history) / len(self.inference_time_history)
            if avg_time > 0.1:  # >100ms is slow
                self.send_alert(f"PERFORMANCE_ISSUE: Average inference time {avg_time:.3f}s > 0.1s")

        # Check safety status
        if not self.safety_status:
            self.send_alert("SAFETY_DISABLED: VLA safety monitoring is disabled")

        # Log system status
        status_msg = f"VLA System Status - "
        if self.confidence_history:
            avg_conf = sum(self.confidence_history[-10:]) / len(self.confidence_history[-10:])
            status_msg += f"Conf: {avg_conf:.2f}, "
        if self.inference_time_history:
            avg_time = sum(self.inference_time_history[-10:]) / len(self.inference_time_history[-10:])
            status_msg += f"Time: {avg_time:.3f}s, "
        status_msg += f"Safe: {self.safety_status}"

        self.get_logger().info(status_msg)

    def send_alert(self, message):
        """
        Send alert message.
        """
        alert_msg = String()
        alert_msg.data = message
        self.alert_pub.publish(alert_msg)
        self.get_logger().warn(f"VLA Alert: {message}")

def main(args=None):
    rclpy.init(args=args)
    node = VLAMonitorNode()

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

---

## 4.2.8 Hands-On Exercise: Complete VLA-ROS2 Integration

### Exercise Objectives

Build a complete VLA-ROS2 system that:
1. Takes natural language commands and camera images
2. Processes them through a VLA model
3. Executes actions on a simulated humanoid
4. Includes monitoring and safety features

### Step 1: Create Complete Package Structure

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python vla_ros2_integration
cd vla_ros2_integration

# Create directory structure
mkdir -p vla_ros2_integration/{nodes,utils,models}
mkdir -p launch config
```

### Step 2: Create Main Integration Node

```python
# vla_ros2_integration/nodes/vla_integration_node.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import numpy as np

from vla_interfaces.action import VLAManipulation

class VLAIntegrationNode(Node):
    """
    Complete VLA-ROS2 integration node.
    """

    def __init__(self):
        super().__init__('vla_integration_node')

        # Initialize VLA model
        self.vla_model = self.load_vla_model()
        self.vla_model.eval()

        # ROS interfaces
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.command_sub = self.create_subscription(
            String, '/vla_command', self.command_callback, 10)

        # Publishers
        self.action_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.confidence_pub = self.create_publisher(Float32, '/vla_confidence', 10)

        # Action server
        self.action_server = ActionServer(
            self, VLAManipulation, 'vla_execute', self.action_execute_callback)

        # State
        self.current_image = None
        self.current_command = None

        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("VLA Integration Node ready")

    def load_vla_model(self):
        """Load VLA model."""
        from vla_humanoid_control.models.simple_vla import SimpleVLA
        model = SimpleVLA()
        return model

    def image_callback(self, msg):
        """Process camera images."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            import cv2
            resized_image = cv2.resize(cv_image, (224, 224))
            self.current_image = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")

    def command_callback(self, msg):
        """Process language commands."""
        self.current_command = msg.data

    def control_loop(self):
        """Main control loop."""
        if self.current_image is None or self.current_command is None:
            return

        try:
            # Prepare inputs
            image_tensor = self.current_image.unsqueeze(0)
            tokenizer = SimpleTokenizer()
            command_tokens = tokenizer.tokenize(self.current_command).unsqueeze(0)

            # VLA inference
            with torch.no_grad():
                output = self.vla_model(image_tensor, command_tokens)

            actions = output['actions'].squeeze(0).numpy()
            confidence = output['confidence'].squeeze(0).item()

            # Publish confidence
            conf_msg = Float32()
            conf_msg.data = confidence
            self.confidence_pub.publish(conf_msg)

            # Execute action if confidence is sufficient
            if confidence > 0.6:
                cmd = Twist()
                cmd.linear.x = float(actions[0]) * 0.5
                cmd.linear.y = float(actions[1]) * 0.5
                cmd.angular.z = float(actions[5]) * 0.5
                self.action_pub.publish(cmd)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")

    def action_execute_callback(self, goal_handle):
        """Execute VLA action via action server."""
        command = goal_handle.request.command_text
        self.get_logger().info(f"Executing VLA action: {command}")

        # Set command and wait for execution
        self.current_command = command

        # Simulate execution
        import time
        time.sleep(2.0)

        # Return result
        result = VLAManipulation.Result()
        result.success = True
        result.confidence_score = 0.8  # Simulated confidence
        goal_handle.succeed()
        return result

class SimpleTokenizer:
    """Simple tokenizer (same as previous examples)."""
    def __init__(self):
        self.vocab = {
            'go': 1, 'move': 2, 'forward': 3, 'backward': 4,
            'left': 5, 'right': 6, 'turn': 7, 'stop': 8,
            'pick': 9, 'place': 10, 'grasp': 11, 'release': 12,
            'the': 13, 'a': 14, 'an': 15, 'and': 16, 'or': 17,
            'up': 18, 'down': 19, 'to': 20, 'at': 21, 'on': 22,
            'red': 23, 'blue': 24, 'green': 25, 'cup': 26, 'box': 27,
            'table': 28, 'chair': 29, 'kitchen': 30, 'room': 31
        }
        self.unk_token = 0

    def tokenize(self, text):
        words = text.lower().split()
        token_ids = [self.vocab.get(word, self.unk_token) for word in words]
        if len(token_ids) < 10:
            token_ids.extend([0] * (10 - len(token_ids)))
        else:
            token_ids = token_ids[:10]
        return torch.tensor(token_ids, dtype=torch.long)

def main(args=None):
    rclpy.init(args=args)
    node = VLAIntegrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 3: Create Launch File

```python
# launch/vla_integration.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),

        Node(
            package='vla_ros2_integration',
            executable='vla_integration_node',
            name='vla_integration',
            parameters=[{'use_sim_time': use_sim_time}],
            remappings=[
                ('/camera/rgb/image_raw', '/camera/image_raw'),
                ('/vla_command', '/vla/command'),
                ('/cmd_vel', '/robot/cmd_vel')
            ]
        )
    ])
```

### Step 4: Test the Integration

```bash
# Build the package
cd ~/ros2_ws
colcon build --packages-select vla_ros2_integration vla_interfaces
source install/setup.bash

# Run the integration node
ros2 run vla_ros2_integration vla_integration_node

# In another terminal, send test commands
ros2 topic pub /vla_command std_msgs/String "data: 'move forward slowly'"

# Monitor confidence
ros2 topic echo /vla_confidence

# Check available actions
ros2 action list
ros2 action send_goal /vla_execute vla_interfaces/action/VLAManipulation "{command_text: 'pick up the object'}"
```

---

## Summary

In this chapter, you learned:

✅ **Custom message types**: Designing multimodal communication for VLA systems
✅ **Action servers**: Implementing VLA-based action execution with feedback
✅ **Distributed architecture**: Separating perception, inference, and control
✅ **Navigation integration**: Connecting VLA with ROS 2 navigation stack
✅ **Performance optimization**: GPU acceleration and batch processing
✅ **System monitoring**: Tracking VLA system health and performance
✅ **Complete integration**: Building end-to-end VLA-ROS2 systems

### Key Takeaways

1. **Modular design**: Separate VLA components into dedicated nodes for scalability
2. **Real-time performance**: Optimize inference and use appropriate QoS settings
3. **Safety first**: Implement confidence thresholds and safety monitoring
4. **Robust communication**: Use actions for complex tasks, topics for streaming data
5. **Monitoring**: Track system performance and generate alerts for issues
6. **Integration**: Connect VLA with existing ROS 2 navigation and manipulation stacks

---

## Next Steps

In **Chapter 4.3: Humanoid Control with VLA**, you'll learn:
- Advanced humanoid-specific VLA implementations
- Whole-body control with VLA models
- Balance-aware VLA for bipedal robots
- Humanoid manipulation and locomotion coordination

**Recommended Practice:**
1. Implement the complete VLA-ROS2 system with your humanoid robot
2. Optimize inference performance for real-time operation
3. Integrate with Nav2 for language-guided navigation
4. Test safety monitoring and confidence thresholds

---

## Additional Resources

### ROS 2 Documentation
- [ROS 2 Actions](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Using-Actions-In-Different-Languages.html)
- [Custom Message Definitions](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html)
- [ROS 2 Launch Files](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Launch-Main.html)

### Performance Optimization
- [TensorRT for PyTorch](https://pytorch.org/tutorials/recipes/tensorrt.html)
- [ROS 2 QoS Settings](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html)
- [Multi-threaded Executors](https://docs.ros.org/en/humble/How-To-Guides/Using-Executors.html)

### VLA Integration Patterns
- [Robot Operating System (ROS) for AI](https://arxiv.org/abs/2306.01473)
- [Multimodal Robotics with Deep Learning](https://arxiv.org/abs/2402.15508)

---

**End of Chapter 4.2: VLA-ROS2 Integration**
