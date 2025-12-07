/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro',
        'table-of-contents',  // Added table of contents page
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-01-ros2/module-01-ros2-index',
        {
          type: 'doc',
          id: 'module-01-ros2/chapter-01-01-architecture',
          label: 'Chapter 1: ROS 2 Architecture & Core Concepts',
        },
        {
          type: 'doc',
          id: 'module-01-ros2/chapter-01-02-nodes-topics-services',
          label: 'Chapter 2: Nodes, Topics & Services',
        },
        {
          type: 'doc',
          id: 'module-01-ros2/chapter-01-03-workspaces-packages',
          label: 'Chapter 3: Workspaces and Packages',
        },
        {
          type: 'doc',
          id: 'module-01-ros2/chapter-01-04-urdf-robot-description',
          label: 'Chapter 4: URDF - Robot Description Language',
        },
        {
          type: 'doc',
          id: 'module-01-ros2/chapter-01-05-launch-files',
          label: 'Chapter 5: Launch Files - Coordinating Complex Robot Systems',
        },
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-02-digital-twin/module-02-digital-twin-index',
        {
          type: 'doc',
          id: 'module-02-digital-twin/chapter-02-01-simulation-fundamentals',
          label: 'Chapter 1: Simulation Fundamentals & Physics Engines',
        },
        {
          type: 'doc',
          id: 'module-02-digital-twin/chapter-02-02-gazebo-basics',
          label: 'Chapter 2: Gazebo Basics - Creating Virtual Worlds',
        },
        {
          type: 'doc',
          id: 'module-02-digital-twin/chapter-02-03-unity-visualization',
          label: 'Chapter 3: Unity Visualization - Advanced 3D Rendering',
        },
        {
          type: 'doc',
          id: 'module-02-digital-twin/chapter-02-04-sim-physical-connection',
          label: 'Chapter 4: Connecting Simulation to Physical Robots',
        },
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module-03-isaac/module-03-isaac-index',
        {
          type: 'doc',
          id: 'module-03-isaac/chapter-03-01-isaac-sim-fundamentals',
          label: 'Chapter 1: Isaac Sim Fundamentals - NVIDIA\'s Robotics Simulation',
        },
        {
          type: 'doc',
          id: 'module-03-isaac/chapter-03-02-isaac-ros-bridge',
          label: 'Chapter 2: Isaac ROS Bridge - Connecting AI to Robotics',
        },
        {
          type: 'doc',
          id: 'module-03-isaac/chapter-03-03-robot-control-with-isaac',
          label: 'Chapter 3: Robot Control with Isaac - Advanced AI Integration',
        },
        {
          type: 'doc',
          id: 'module-03-isaac/chapter-03-04-physical-ai-concepts',
          label: 'Chapter 4: Physical AI Concepts - Intelligence in Motion',
        },
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-04-vla/module-04-vla-index',
        {
          type: 'doc',
          id: 'module-04-vla/chapter-04-01-vla-fundamentals',
          label: 'Chapter 1: VLA Fundamentals - AI That Sees, Understands & Acts',
        },
        {
          type: 'doc',
          id: 'module-04-vla/chapter-04-02-vla-ros2-integration',
          label: 'Chapter 2: VLA-ROS2 Integration - Connecting AI Models to Robotics',
        },
        {
          type: 'doc',
          id: 'module-04-vla/chapter-04-03-humanoid-control-with-vla',
          label: 'Chapter 3: Humanoid Control with VLA - Advanced AI Control Systems',
        },
        {
          type: 'doc',
          id: 'module-04-vla/chapter-04-04-capstone-project',
          label: 'Chapter 4: Capstone Project - Building an Intelligent Humanoid Robot',
        },
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/index',
      ],
      collapsed: true,
    },
  ],
};

export default sidebars;