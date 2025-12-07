---
id: chapter-02-03-unity-visualization
title: "Unity Visualization"
sidebar_label: "Chapter 3: Unity Visualization"
description: "Understanding Unity as a visualization platform for humanoid robots and integrating with ROS 2"
keywords:
  - Unity
  - Visualization
  - ROS 2 Integration
  - Humanoid Robots
prerequisites:
  - chapter-01-01-architecture
  - chapter-01-02-nodes-topics-services
  - chapter-02-01-simulation-fundamentals
---


# Chapter 3 - Unity Visualization

## Learning Objectives

- Understand Unity as a visualization platform for humanoid robots
- Integrate Unity with ROS 2 for real-time robot visualization
- Create high-fidelity 3D environments for human-robot interaction
- Implement sensor visualization in Unity
- Design interactive interfaces for robot monitoring and control

## Prerequisites

- Completed Module 1: The Robotic Nervous System (ROS 2)
- Completed Module 2 Chapters 1-2: Simulation fundamentals and Gazebo basics
- Ubuntu 22.04 LTS with ROS 2 Humble installed
- Unity 2021.3 LTS or newer
- Basic knowledge of C# programming
- Understanding of 3D graphics and rendering concepts

## Introduction

Unity provides a powerful platform for visualizing humanoid robots with high-fidelity graphics and interactive capabilities. While Gazebo excels at physics simulation, Unity's strength lies in photorealistic rendering, user interface design, and human-robot interaction scenarios. This chapter explores how to use Unity as a visualization layer for ROS 2-based humanoid robotics systems, enabling compelling demonstrations and user interactions.

Unity's real-time 3D engine allows developers to create immersive experiences where users can interact with robots in realistic environments, making it ideal for training, demonstrations, and user studies.

### Why Unity for Robotics Visualization

Unity offers several advantages for robotics visualization:
- Photorealistic rendering with advanced lighting and materials
- Cross-platform deployment (desktop, mobile, VR/AR)
- Rich ecosystem of assets and tools
- Strong UI/UX capabilities for control interfaces
- Excellent performance for real-time rendering

### Real-world Applications

- Robot teleoperation interfaces with immersive visualization
- Training simulations for human-robot collaboration
- Virtual showrooms for robot demonstrations
- Human-robot interaction research platforms

### What You'll Build by the End

By completing this chapter, you will create:
- Unity project integrated with ROS 2
- High-fidelity humanoid robot visualization
- Interactive control interface
- Real-time sensor data visualization

## Core Concepts

### Unity-ROS Integration Architecture

Unity-ROS integration involves:
- ROS-TCP-Connector for communication
- Message serialization and deserialization
- Transform synchronization (TF)
- Real-time data streaming

### Rendering Pipeline

Unity's rendering pipeline includes:
- Universal Render Pipeline (URP) for performance
- High Definition Render Pipeline (HDRP) for quality
- Shader graphs for custom materials
- Post-processing effects

### Human-Robot Interaction Design

Effective HRI visualization requires:
- Clear robot state indication
- Intuitive control interfaces
- Spatial awareness visualization
- Safety zone representation

## Hands-On Tutorial

### Step 1: Install Unity and ROS-TCP-Connector

First, install Unity and set up the ROS-TCP-Connector:

```bash
# Install Unity Hub (if not already installed)
# Download from https://unity.com/download

# After installing Unity 2021.3 LTS through Unity Hub, create a new 3D project

# Install ROS-TCP-Endpoint on your ROS 2 system
cd ~/unity_ros_ws/src
git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
cd ~/unity_ros_ws
colcon build --packages-select ros_tcp_endpoint
source install/setup.bash
```

### Step 2: Set Up Unity Project

Create a new Unity project and install the ROS-TCP-Connector package:

1. Open Unity Hub and create a new 3D project named "HumanoidRobotViz"
2. Open the Package Manager (Window > Package Manager)
3. Click the "+" button and select "Add package from git URL"
4. Enter: `https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-tcp-connector`

### Step 3: Configure ROS-TCP Connection

Create a ROS Settings configuration in Unity:

```csharp
// Assets/Scripts/ROSConnection.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using RosMessageTypes.Sensor;

public class ROSConnection : MonoBehaviour
{
    // ROS connection settings
    private ROSConnection rosConnection;
    private string rosIPAddress = "127.0.0.1";
    private int rosPort = 10000;

    // Topic names
    private string jointStateTopic = "/joint_states";
    private string cmdVelTopic = "/cmd_vel";
    private string cameraTopic = "/camera/rgb/image_raw";
    private string imuTopic = "/imu/data";

    void Start()
    {
        // Get ROS connection instance
        rosConnection = ROSConnection.GetOrCreateInstance();
        rosConnection.Connect(rosIPAddress, rosPort);

        // Subscribe to ROS topics
        rosConnection.Subscribe<RosMessageTypes.Sensor.JointStateMsg>(
            jointStateTopic, ReceiveJointState);
        rosConnection.Subscribe<RosMessageTypes.Geometry.TwistMsg>(
            cmdVelTopic, ReceiveCmdVel);
        rosConnection.Subscribe<RosMessageTypes.Sensor.ImuMsg>(
            imuTopic, ReceiveImu);

        Debug.Log($"Connected to ROS at {rosIPAddress}:{rosPort}");
    }

    void ReceiveJointState(RosMessageTypes.Sensor.JointStateMsg jointState)
    {
        // Process joint state messages
        Debug.Log($"Received joint state with {jointState.name.Length} joints");

        // Update robot visualization based on joint states
        UpdateRobotJoints(jointState);
    }

    void ReceiveCmdVel(RosMessageTypes.Geometry.TwistMsg cmdVel)
    {
        // Process velocity commands
        float linearX = (float)cmdVel.linear.x;
        float angularZ = (float)cmdVel.angular.z;

        Debug.Log($"Cmd Vel: linear={linearX}, angular={angularZ}");

        // Visualize command velocity
        VisualizeCmdVel(linearX, angularZ);
    }

    void ReceiveImu(RosMessageTypes.Sensor.ImuMsg imu)
    {
        // Process IMU data
        Vector3 orientation = new Vector3(
            (float)imu.orientation.x,
            (float)imu.orientation.y,
            (float)imu.orientation.z
        );

        Debug.Log($"IMU orientation: {orientation}");
    }

    void UpdateRobotJoints(RosMessageTypes.Sensor.JointStateMsg jointState)
    {
        // Find robot articulation body and update joint angles
        // This will be implemented in the robot controller script
    }

    void VisualizeCmdVel(float linear, float angular)
    {
        // Create visual representation of command velocity
        // Could be arrows or other indicators
    }

    void OnDestroy()
    {
        // Clean up ROS connection
        if (rosConnection != null)
        {
            rosConnection.Disconnect();
        }
    }
}
```

### Step 4: Create Humanoid Robot Visualization

Create a script to control the humanoid robot model in Unity:

```csharp
// Assets/Scripts/HumanoidRobotController.cs
using UnityEngine;
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class HumanoidRobotController : MonoBehaviour
{
    // Robot model components
    [Header("Robot Components")]
    public Transform baseLink;
    public ArticulationBody rootArticulation;

    // Joint mappings
    private Dictionary<string, ArticulationBody> jointMap = new Dictionary<string, ArticulationBody>();
    private Dictionary<string, Transform> transformMap = new Dictionary<string, Transform>();

    // Visual indicators
    [Header("Visual Indicators")]
    public GameObject balanceIndicator;
    public LineRenderer trajectoryRenderer;
    public GameObject sensorVisualizer;

    // Robot state
    private Vector3 currentPosition;
    private Quaternion currentOrientation;
    private float balanceScore = 1.0f;

    void Start()
    {
        // Initialize joint mappings
        InitializeJointMappings();

        // Set up visual indicators
        SetupVisualIndicators();

        Debug.Log($"Humanoid robot controller initialized with {jointMap.Count} joints");
    }

    void InitializeJointMappings()
    {
        // Find all articulation bodies in the robot
        ArticulationBody[] articulationBodies = GetComponentsInChildren<ArticulationBody>();

        foreach (ArticulationBody body in articulationBodies)
        {
            string jointName = body.name;
            jointMap[jointName] = body;
            transformMap[jointName] = body.transform;

            Debug.Log($"Mapped joint: {jointName}");
        }
    }

    void SetupVisualIndicators()
    {
        // Create balance indicator if not assigned
        if (balanceIndicator == null)
        {
            balanceIndicator = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            balanceIndicator.transform.localScale = new Vector3(0.2f, 0.2f, 0.2f);
            balanceIndicator.GetComponent<Renderer>().material.color = Color.green;
        }

        // Set up trajectory renderer
        if (trajectoryRenderer == null)
        {
            GameObject trajectoryObj = new GameObject("TrajectoryRenderer");
            trajectoryRenderer = trajectoryObj.AddComponent<LineRenderer>();
            trajectoryRenderer.startWidth = 0.05f;
            trajectoryRenderer.endWidth = 0.05f;
            trajectoryRenderer.material = new Material(Shader.Find("Sprites/Default"));
            trajectoryRenderer.startColor = Color.blue;
            trajectoryRenderer.endColor = Color.cyan;
        }
    }

    public void UpdateJointStates(JointStateMsg jointState)
    {
        // Update each joint based on ROS joint state message
        for (int i = 0; i < jointState.name.Length; i++)
        {
            string jointName = jointState.name[i];
            float position = (float)jointState.position[i];

            if (jointMap.ContainsKey(jointName))
            {
                UpdateJointPosition(jointMap[jointName], position);
            }
        }

        // Update balance indicator
        UpdateBalanceVisualization();
    }

    void UpdateJointPosition(ArticulationBody joint, float targetPosition)
    {
        // Set articulation body target based on joint type
        ArticulationDrive drive = joint.xDrive;
        drive.target = targetPosition * Mathf.Rad2Deg; // Convert to degrees
        joint.xDrive = drive;
    }

    void UpdateBalanceVisualization()
    {
        // Calculate balance score based on robot orientation
        Vector3 up = transform.up;
        balanceScore = Vector3.Dot(up, Vector3.up);

        // Update balance indicator color
        if (balanceIndicator != null)
        {
            Color indicatorColor = Color.Lerp(Color.red, Color.green, balanceScore);
            balanceIndicator.GetComponent<Renderer>().material.color = indicatorColor;

            // Position indicator relative to robot
            balanceIndicator.transform.position = baseLink.position + Vector3.up * 2.0f;
        }
    }

    public void UpdateTrajectory(Vector3[] waypoints)
    {
        // Update trajectory visualization
        if (trajectoryRenderer != null && waypoints.Length > 0)
        {
            trajectoryRenderer.positionCount = waypoints.Length;
            trajectoryRenderer.SetPositions(waypoints);
        }
    }

    public void VisualizeSensorData(string sensorType, Vector3 position, Quaternion rotation)
    {
        // Create sensor visualization
        if (sensorVisualizer == null)
        {
            sensorVisualizer = new GameObject($"Sensor_{sensorType}");
        }

        sensorVisualizer.transform.position = position;
        sensorVisualizer.transform.rotation = rotation;
    }

    void Update()
    {
        // Update current position and orientation
        if (baseLink != null)
        {
            currentPosition = baseLink.position;
            currentOrientation = baseLink.rotation;
        }

        // Update visual indicators every frame
        if (balanceIndicator != null)
        {
            UpdateBalanceVisualization();
        }
    }

    // Public methods for external control
    public void SetRobotPose(Vector3 position, Quaternion rotation)
    {
        if (baseLink != null)
        {
            baseLink.position = position;
            baseLink.rotation = rotation;
        }
    }

    public Vector3 GetCurrentPosition()
    {
        return currentPosition;
    }

    public Quaternion GetCurrentOrientation()
    {
        return currentOrientation;
    }

    public float GetBalanceScore()
    {
        return balanceScore;
    }
}
```

### Step 5: Create Camera Visualization

Create a script to visualize camera feeds from ROS:

```csharp
// Assets/Scripts/CameraVisualization.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class CameraVisualization : MonoBehaviour
{
    [Header("Camera Settings")]
    public string cameraTopicName = "/camera/rgb/image_raw";
    public Material displayMaterial;
    public Renderer displayRenderer;

    // Texture for camera feed
    private Texture2D cameraTexture;
    private int imageWidth = 640;
    private int imageHeight = 480;

    // ROS connection
    private ROSConnection rosConnection;

    void Start()
    {
        // Get ROS connection
        rosConnection = ROSConnection.GetOrCreateInstance();

        // Subscribe to camera topic
        rosConnection.Subscribe<ImageMsg>(cameraTopicName, ReceiveCameraImage);

        // Create texture for camera feed
        cameraTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        // Apply texture to material
        if (displayMaterial != null)
        {
            displayMaterial.mainTexture = cameraTexture;
        }

        if (displayRenderer != null)
        {
            displayRenderer.material = displayMaterial;
        }

        Debug.Log($"Camera visualization subscribed to {cameraTopicName}");
    }

    void ReceiveCameraImage(ImageMsg imageMsg)
    {
        // Update image dimensions if needed
        if (imageMsg.width != imageWidth || imageMsg.height != imageHeight)
        {
            imageWidth = (int)imageMsg.width;
            imageHeight = (int)imageMsg.height;
            cameraTexture.Reinitialize(imageWidth, imageHeight);
        }

        // Convert ROS image to Unity texture
        byte[] imageData = imageMsg.data;

        // Handle different encoding formats
        if (imageMsg.encoding == "rgb8")
        {
            cameraTexture.LoadRawTextureData(imageData);
            cameraTexture.Apply();
        }
        else if (imageMsg.encoding == "bgr8")
        {
            // Convert BGR to RGB
            byte[] rgbData = ConvertBGRtoRGB(imageData);
            cameraTexture.LoadRawTextureData(rgbData);
            cameraTexture.Apply();
        }
    }

    byte[] ConvertBGRtoRGB(byte[] bgrData)
    {
        byte[] rgbData = new byte[bgrData.Length];
        for (int i = 0; i < bgrData.Length; i += 3)
        {
            rgbData[i] = bgrData[i + 2];     // R = B
            rgbData[i + 1] = bgrData[i + 1]; // G = G
            rgbData[i + 2] = bgrData[i];     // B = R
        }
        return rgbData;
    }

    void OnDestroy()
    {
        // Clean up texture
        if (cameraTexture != null)
        {
            Destroy(cameraTexture);
        }
    }
}
```

### Step 6: Create Interactive Control Interface

Create a UI for robot control:

```csharp
// Assets/Scripts/RobotControlUI.cs
using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class RobotControlUI : MonoBehaviour
{
    [Header("UI Elements")]
    public Slider linearVelocitySlider;
    public Slider angularVelocitySlider;
    public Text linearVelocityText;
    public Text angularVelocityText;
    public Button stopButton;
    public Text statusText;

    [Header("Control Settings")]
    public string cmdVelTopic = "/cmd_vel";
    public float maxLinearVelocity = 1.0f;
    public float maxAngularVelocity = 1.0f;

    // ROS connection
    private ROSConnection rosConnection;

    // Current velocities
    private float currentLinearVel = 0f;
    private float currentAngularVel = 0f;

    void Start()
    {
        // Get ROS connection
        rosConnection = ROSConnection.GetOrCreateInstance();

        // Register publisher for cmd_vel
        rosConnection.RegisterPublisher<TwistMsg>(cmdVelTopic);

        // Set up UI listeners
        if (linearVelocitySlider != null)
        {
            linearVelocitySlider.minValue = -maxLinearVelocity;
            linearVelocitySlider.maxValue = maxLinearVelocity;
            linearVelocitySlider.value = 0;
            linearVelocitySlider.onValueChanged.AddListener(OnLinearVelocityChanged);
        }

        if (angularVelocitySlider != null)
        {
            angularVelocitySlider.minValue = -maxAngularVelocity;
            angularVelocitySlider.maxValue = maxAngularVelocity;
            angularVelocitySlider.value = 0;
            angularVelocitySlider.onValueChanged.AddListener(OnAngularVelocityChanged);
        }

        if (stopButton != null)
        {
            stopButton.onClick.AddListener(OnStopButtonClicked);
        }

        UpdateStatusText("Ready");
    }

    void OnLinearVelocityChanged(float value)
    {
        currentLinearVel = value;
        UpdateLinearVelocityText(value);
        PublishCmdVel();
    }

    void OnAngularVelocityChanged(float value)
    {
        currentAngularVel = value;
        UpdateAngularVelocityText(value);
        PublishCmdVel();
    }

    void OnStopButtonClicked()
    {
        // Reset sliders to zero
        if (linearVelocitySlider != null)
        {
            linearVelocitySlider.value = 0;
        }

        if (angularVelocitySlider != null)
        {
            angularVelocitySlider.value = 0;
        }

        currentLinearVel = 0f;
        currentAngularVel = 0f;

        PublishCmdVel();
        UpdateStatusText("Stopped");
    }

    void PublishCmdVel()
    {
        // Create Twist message
        TwistMsg cmdVel = new TwistMsg
        {
            linear = new Vector3Msg
            {
                x = currentLinearVel,
                y = 0,
                z = 0
            },
            angular = new Vector3Msg
            {
                x = 0,
                y = 0,
                z = currentAngularVel
            }
        };

        // Publish to ROS
        rosConnection.Publish(cmdVelTopic, cmdVel);

        UpdateStatusText($"Publishing: linear={currentLinearVel:F2}, angular={currentAngularVel:F2}");
    }

    void UpdateLinearVelocityText(float value)
    {
        if (linearVelocityText != null)
        {
            linearVelocityText.text = $"Linear: {value:F2} m/s";
        }
    }

    void UpdateAngularVelocityText(float value)
    {
        if (angularVelocityText != null)
        {
            angularVelocityText.text = $"Angular: {value:F2} rad/s";
        }
    }

    void UpdateStatusText(string status)
    {
        if (statusText != null)
        {
            statusText.text = $"Status: {status}";
        }
    }

    void Update()
    {
        // Optional: Add keyboard control
        if (Input.GetKey(KeyCode.W))
        {
            if (linearVelocitySlider != null)
            {
                linearVelocitySlider.value = Mathf.Min(linearVelocitySlider.value + 0.01f, maxLinearVelocity);
            }
        }
        else if (Input.GetKey(KeyCode.S))
        {
            if (linearVelocitySlider != null)
            {
                linearVelocitySlider.value = Mathf.Max(linearVelocitySlider.value - 0.01f, -maxLinearVelocity);
            }
        }

        if (Input.GetKey(KeyCode.A))
        {
            if (angularVelocitySlider != null)
            {
                angularVelocitySlider.value = Mathf.Min(angularVelocitySlider.value + 0.01f, maxAngularVelocity);
            }
        }
        else if (Input.GetKey(KeyCode.D))
        {
            if (angularVelocitySlider != null)
            {
                angularVelocitySlider.value = Mathf.Max(angularVelocitySlider.value - 0.01f, -maxAngularVelocity);
            }
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            OnStopButtonClicked();
        }
    }
}
```

### Step 7: Launch ROS-TCP-Endpoint

Start the ROS-TCP-Endpoint to connect Unity with ROS 2:

```bash
# Terminal 1: Start ROS-TCP-Endpoint
cd ~/unity_ros_ws
source install/setup.bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0

# Terminal 2: Start your robot (simulation or real)
source ~/robot_ws/install/setup.bash
ros2 launch robot_bringup robot.launch.py

# Terminal 3: Verify topics
ros2 topic list
ros2 topic echo /joint_states
```

### Step 8: Test Unity Visualization

1. Open your Unity project
2. Create a scene with:
   - A humanoid robot model (import URDF or create manually)
   - A plane for the ground
   - Lighting and camera
3. Attach the scripts to appropriate GameObjects:
   - ROSConnection to a GameObject named "ROSManager"
   - HumanoidRobotController to your robot model
   - RobotControlUI to a Canvas with UI elements
4. Press Play in Unity
5. Verify connection to ROS and data flow

Expected results: You should see the robot visualized in Unity, updating in real-time based on ROS data, with an interactive control interface.

## Troubleshooting

### Common Error 1: Connection Failed to ROS-TCP-Endpoint
**Cause**: ROS-TCP-Endpoint not running or wrong IP address
**Solution**: Verify endpoint is running and check IP configuration
**Prevention Tips**: Use localhost (127.0.0.1) for local testing

### Common Error 2: Robot Model Not Updating
**Cause**: Joint name mismatch between ROS and Unity
**Solution**: Verify joint names match exactly between systems
**Prevention Tips**: Use consistent naming conventions

### Common Error 3: Poor Rendering Performance
**Cause**: Complex models or inefficient materials
**Solution**: Optimize meshes and use appropriate render pipeline
**Prevention Tips**: Use LOD (Level of Detail) for complex models

## Key Takeaways

- Unity provides high-fidelity visualization for humanoid robots
- ROS-TCP-Connector enables seamless Unity-ROS integration
- Interactive interfaces enhance robot control and monitoring
- Visual feedback improves human-robot interaction
- Cross-platform deployment extends accessibility

## Additional Resources

- [Unity Robotics Hub](https://github.com/Unity-Technologies/Unity-Robotics-Hub)
- [ROS-TCP-Connector Documentation](https://github.com/Unity-Technologies/ROS-TCP-Connector)
- [Unity Learn Tutorials](https://learn.unity.com/)
- [URDF Importer for Unity](https://github.com/Unity-Technologies/URDF-Importer)

## Self-Assessment

1. How does Unity-ROS integration differ from Gazebo-ROS integration?
2. What are the advantages of using Unity for robot visualization?
3. How do you synchronize robot joint states between ROS and Unity?
4. What rendering pipeline should you choose for mobile deployment?
5. How would you implement VR control for a humanoid robot in Unity?

<ChapterNavigation
  previous={{
    permalink: '/docs/module-02-digital-twin/chapter-02-02-gazebo-basics',
    title: '2.2 Gazebo Basics'
  }}
  next={{
    permalink: '/docs/module-02-digital-twin/chapter-02-04-sim-physical-connection',
    title: '2.4 Simulation-Physical Connection'
  }}
/>
