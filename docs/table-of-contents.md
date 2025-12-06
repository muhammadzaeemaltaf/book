---
id: table-of-contents
title: "Table of Contents - Physical AI & Humanoid Robotics Textbook"
sidebar_label: "Table of Contents"
description: "Complete structure and organization of the Physical AI & Humanoid Robotics textbook"
keywords:
  - Table of Contents
  - Textbook Structure
  - Learning Path
  - Physical AI
  - Humanoid Robotics
prerequisites: []
---

<BreadcrumbNavigation
  items={[
    { title: 'Textbook', url: '/docs/intro' },
    { title: 'Table of Contents', url: '/docs/table-of-contents' }
  ]}
/>

# Table of Contents: Physical AI & Humanoid Robotics Textbook

## Complete Textbook Structure

This comprehensive textbook is organized into four progressive modules, each building upon the previous to provide a complete understanding of Physical AI and Humanoid Robotics.

## Module 1: The Robotic Nervous System (ROS 2)

- **Module Overview**: [Module 1: The Robotic Nervous System (ROS 2)](/docs/module-01-ros2/index)
- **Chapter 1**: [ROS 2 Architecture & Core Concepts](/docs/module-01-ros2/chapter-01-01-architecture)
  - Understanding fundamental ROS 2 architecture
  - Core concepts of distributed robotic systems
- **Chapter 2**: [Nodes, Topics, Services and Actions](/docs/module-01-ros2/chapter-01-02-nodes-topics-services)
  - Deep dive into ROS 2 communication mechanisms
  - Publisher-subscriber and service-client patterns
- **Chapter 3**: [Workspaces and Packages](/docs/module-01-ros2/chapter-01-03-workspaces-packages)
  - Creating and managing ROS 2 development environments
  - Building custom packages and launch files
- **Chapter 4**: [URDF & Robot Description](/docs/module-01-ros2/chapter-01-04-urdf-robot-description)
  - Modeling robots using Unified Robot Description Format
  - Defining robot structure and kinematics
- **Chapter 5**: [Launch Files - Coordinating Complex Robot Systems](/docs/module-01-ros2/chapter-01-05-launch-files)
  - Coordinating multi-node robot systems with launch files
  - Advanced system orchestration techniques

## Module 2: The Digital Twin (Gazebo & Unity)

- **Module Overview**: [Module 2: The Digital Twin (Gazebo & Unity)](/docs/module-02-digital-twin/index)
- **Chapter 1**: [Simulation Fundamentals](/docs/module-02-digital-twin/chapter-02-01-simulation-fundamentals)
  - Core principles of physics simulation in robotics
  - Understanding digital twins and their applications
- **Chapter 2**: [Gazebo Basics](/docs/module-02-digital-twin/chapter-02-02-gazebo-basics)
  - Physics simulation with Gazebo
  - Creating and configuring robot models for simulation
- **Chapter 3**: [Unity Visualization](/docs/module-02-digital-twin/chapter-02-03-unity-visualization)
  - Unity as a visualization platform for humanoid robots
  - Integrating Unity with ROS 2 for real-time visualization
- **Chapter 4**: [Simulation-Physical Connection](/docs/module-02-digital-twin/chapter-02-04-sim-physical-connection)
  - Connecting simulation to physical reality
  - Validation techniques and sim-to-real transfer

## Module 3: The AI-Robot Brain (NVIDIA Isaac™)

- **Module Overview**: [Module 3: The AI-Robot Brain (NVIDIA Isaac™)](/docs/module-03-isaac/index)
- **Chapter 1**: [Isaac Sim Fundamentals](/docs/module-03-isaac/chapter-03-01-isaac-sim-fundamentals)
  - NVIDIA Isaac Sim architecture and capabilities
  - Setting up Isaac Sim for robotic simulation
- **Chapter 2**: [Isaac ROS Bridge](/docs/module-03-isaac/chapter-03-02-isaac-ros-bridge)
  - Isaac ROS framework and hardware-accelerated perception
  - VSLAM and navigation with Nav2
- **Chapter 3**: [Robot Control with Isaac](/docs/module-03-isaac/chapter-03-03-robot-control-with-isaac)
  - Advanced navigation and manipulation with Isaac
  - Humanoid-specific path planning algorithms
- **Chapter 4**: [Physical AI Concepts](/docs/module-03-isaac/chapter-03-04-physical-ai-concepts)
  - Embodied intelligence and sensor-motor integration
  - World models and sim-to-real transfer

## Module 4: Vision-Language-Action (VLA)

- **Module Overview**: [Module 4: Vision-Language-Action (VLA)](/docs/module-04-vla/index)
- **Chapter 1**: [VLA Fundamentals](/docs/module-04-vla/chapter-04-01-vla-fundamentals)
  - Vision-Language-Action models for humanoid robotics
  - Multimodal architectures and embodied language models
- **Chapter 2**: [VLA-ROS2 Integration](/docs/module-04-vla/chapter-04-02-vla-ros2-integration)
  - Integrating VLA models with ROS 2
  - Multimodal action servers and distributed inference
- **Chapter 3**: [Humanoid Control with VLA](/docs/module-04-vla/chapter-04-03-humanoid-control-with-vla)
  - Advanced VLA implementations for humanoid robots
  - Whole-body control and balance-aware systems
- **Chapter 4**: [Capstone Project](/docs/module-04-vla/chapter-04-04-capstone-project)
  - Complete integration project combining all concepts
  - Building a full humanoid VLA system

## Learning Path Recommendations

The textbook is designed to be studied sequentially, with each module building upon the previous:

1. **Start with Module 1** to establish foundational ROS 2 knowledge
2. **Continue with Module 2** to understand simulation environments
3. **Advance to Module 3** to learn AI integration with NVIDIA Isaac
4. **Complete with Module 4** to master Vision-Language-Action systems

Each chapter includes hands-on exercises and practical examples that reinforce theoretical concepts with real-world implementation.

## Prerequisites

- Basic Python programming skills
- Understanding of Linux command line operations
- Ubuntu 22.04 LTS with ROS 2 Humble installed
- Familiarity with 3D visualization concepts (helpful for simulation modules)
- Understanding of LLMs and transformers (helpful for VLA module)

## Estimated Completion Time

- **Module 1**: 8-10 hours
- **Module 2**: 6-8 hours
- **Module 3**: 8-10 hours
- **Module 4**: 10-12 hours
- **Total**: 32-40 hours of study and hands-on practice

<TableOfContents
  modules={[
    {
      title: "Module 1: The Robotic Nervous System (ROS 2)",
      description: "Foundation of robotic systems with ROS 2 architecture",
      chapters: [
        {
          title: "ROS 2 Architecture & Core Concepts",
          url: "/docs/module-01-ros2/chapter-01-01-architecture"
        },
        {
          title: "Nodes, Topics, Services and Actions",
          url: "/docs/module-01-ros2/chapter-01-02-nodes-topics-services"
        },
        {
          title: "Workspaces and Packages",
          url: "/docs/module-01-ros2/chapter-01-03-workspaces-packages"
        },
        {
          title: "URDF & Robot Description",
          url: "/docs/module-01-ros2/chapter-01-04-urdf-robot-description"
        },
        {
          title: "Launch Files - Coordinating Complex Robot Systems",
          url: "/docs/module-01-ros2/chapter-01-05-launch-files"
        }
      ]
    },
    {
      title: "Module 2: The Digital Twin (Gazebo & Unity)",
      description: "Simulation environments that serve as digital twins of physical robots",
      chapters: [
        {
          title: "Simulation Fundamentals",
          url: "/docs/module-02-digital-twin/chapter-02-01-simulation-fundamentals"
        },
        {
          title: "Gazebo Basics",
          url: "/docs/module-02-digital-twin/chapter-02-02-gazebo-basics"
        },
        {
          title: "Unity Visualization",
          url: "/docs/module-02-digital-twin/chapter-02-03-unity-visualization"
        },
        {
          title: "Simulation-Physical Connection",
          url: "/docs/module-02-digital-twin/chapter-02-04-sim-physical-connection"
        }
      ]
    },
    {
      title: "Module 3: The AI-Robot Brain (NVIDIA Isaac™)",
      description: "NVIDIA Isaac platform for AI-powered robotic applications",
      chapters: [
        {
          title: "Isaac Sim Fundamentals",
          url: "/docs/module-03-isaac/chapter-03-01-isaac-sim-fundamentals"
        },
        {
          title: "Isaac ROS Bridge",
          url: "/docs/module-03-isaac/chapter-03-02-isaac-ros-bridge"
        },
        {
          title: "Robot Control with Isaac",
          url: "/docs/module-03-isaac/chapter-03-03-robot-control-with-isaac"
        },
        {
          title: "Physical AI Concepts",
          url: "/docs/module-03-isaac/chapter-03-04-physical-ai-concepts"
        }
      ]
    },
    {
      title: "Module 4: Vision-Language-Action (VLA)",
      description: "Vision-Language-Action systems for advanced human-robot interaction",
      chapters: [
        {
          title: "VLA Fundamentals",
          url: "/docs/module-04-vla/chapter-04-01-vla-fundamentals"
        },
        {
          title: "VLA-ROS2 Integration",
          url: "/docs/module-04-vla/chapter-04-02-vla-ros2-integration"
        },
        {
          title: "Humanoid Control with VLA",
          url: "/docs/module-04-vla/chapter-04-03-humanoid-control-with-vla"
        },
        {
          title: "Capstone Project",
          url: "/docs/module-04-vla/chapter-04-04-capstone-project"
        }
      ]
    }
  ]}
/>