/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro',
        'intro/intro-index',
        'table-of-contents',  // Added table of contents page
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-01-ros2/module-01-ros2-index',
        'module-01-ros2/chapter-01-01-architecture',
        'module-01-ros2/chapter-01-02-nodes-topics-services',
        'module-01-ros2/chapter-01-03-workspaces-packages',
        'module-01-ros2/chapter-01-04-urdf-robot-description',
        'module-01-ros2/chapter-01-05-launch-files',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-02-digital-twin/module-02-digital-twin-index',
        'module-02-digital-twin/chapter-02-01-simulation-fundamentals',
        'module-02-digital-twin/chapter-02-02-gazebo-basics',
        'module-02-digital-twin/chapter-02-03-unity-visualization',
        'module-02-digital-twin/chapter-02-04-sim-physical-connection',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module-03-isaac/module-03-isaac-index',
        'module-03-isaac/chapter-03-01-isaac-sim-fundamentals',
        'module-03-isaac/chapter-03-02-isaac-ros-bridge',
        'module-03-isaac/chapter-03-03-robot-control-with-isaac',
        'module-03-isaac/chapter-03-04-physical-ai-concepts',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-04-vla/module-04-vla-index',
        'module-04-vla/chapter-04-01-vla-fundamentals',
        'module-04-vla/chapter-04-02-vla-ros2-integration',
        'module-04-vla/chapter-04-03-humanoid-control-with-vla',
        'module-04-vla/chapter-04-04-capstone-project',
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