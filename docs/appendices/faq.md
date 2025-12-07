---
id: faq
title: "Frequently Asked Questions"
sidebar_label: "FAQ"
sidebar_position: 2
description: "Answers to common questions about the Physical AI & Humanoid Robotics textbook"
keywords: [faq, questions, answers, robotics, ai, ros2, isaac]
---

# Frequently Asked Questions

This section addresses common questions about the Physical AI & Humanoid Robotics textbook and its implementation.

## General Questions

### Q: What hardware do I need to follow along with this textbook?
A: The minimum requirements are:
- Ubuntu 22.04 LTS
- 16GB RAM (32GB recommended)
- NVIDIA GPU with at least 8GB VRAM (RTX 3070 or equivalent)
- 100GB free disk space
- ROS 2 Humble Hawksbill installed

For optimal performance with Isaac Sim and VLA models, we recommend:
- RTX 4080 or higher with 16GB+ VRAM
- 32GB+ system RAM
- Multi-core processor (8+ cores)

### Q: Can I use a different OS or ROS distribution?
A: This textbook is specifically designed for Ubuntu 22.04 LTS and ROS 2 Humble Hawksbill. While concepts may apply to other systems, examples and troubleshooting have been validated only on this configuration. Using different versions may result in compatibility issues.

### Q: How long does it take to complete the textbook?
A: With consistent study of 10-15 hours per week, students typically complete the textbook in 12-16 weeks. The capstone project in Module 4 may require additional time depending on your implementation depth.

## ROS 2 Questions

### Q: What's the difference between ROS 1 and ROS 2?
A: ROS 2 addresses key limitations of ROS 1 including:
- Real-time performance capabilities
- Multi-robot system support
- Enhanced security features
- Improved cross-platform compatibility
- DDS-based communication for better scalability

### Q: When should I use topics vs services vs actions?
A: Use topics for continuous data streams (sensor data, robot state), services for request-response interactions (simple commands), and actions for long-running tasks with feedback (navigation, manipulation).

## Isaac Sim Questions

### Q: Why do I need Isaac Sim for robotics development?
A: Isaac Sim provides:
- Photorealistic simulation environments
- Accurate physics modeling
- Synthetic data generation for AI training
- Safe testing of robot behaviors without physical hardware
- Integration with NVIDIA's robotics stack

### Q: How do I transition from simulation to real hardware?
A: The textbook covers simulation-to-reality transfer techniques in Module 2. Key considerations include:
- Domain randomization in simulation
- Sensor noise modeling
- Control parameter tuning
- Hardware-in-the-loop testing

## VLA (Vision-Language-Action) Questions

### Q: What makes VLA different from traditional robotics approaches?
A: VLA models combine:
- Visual perception
- Natural language understanding
- Action generation
- All in a single neural network
This enables more intuitive human-robot interaction compared to traditional task-specific programming.

### Q: What are the computational requirements for VLA models?
A: VLA models require significant computational resources:
- NVIDIA GPU with CUDA support (8GB+ VRAM minimum)
- 16GB+ system RAM for batch processing
- Fast storage (SSD recommended) for data loading
- Models can be run on CPU but with significantly reduced performance

## Troubleshooting Questions

### Q: I'm getting import errors after building my ROS 2 packages. What should I do?
A: This is typically because the workspace hasn't been sourced after building. Run `source install/setup.bash` in your workspace directory after each build.

### Q: My simulation is running very slowly. How can I improve performance?
A: Try these optimizations:
- Reduce simulation resolution in Isaac Sim settings
- Limit the number of active sensors
- Close unnecessary applications to free up system resources
- Check that your GPU drivers are up to date

### Q: How do I verify that my robot control code is working correctly?
A: The textbook includes self-assessment questions at the end of each chapter. Additionally, you can:
- Use RViz2 for visualization of robot state
- Monitor topics with `ros2 topic echo`
- Check logs with `ros2 launch-logs`
- Implement unit tests for critical components

## Development Best Practices

### Q: What are some best practices for robotics software development?
A: Follow these principles:
- Use version control (Git) for all code and configurations
- Write modular, reusable components
- Implement proper error handling and logging
- Test components individually before integration
- Document your code and configurations thoroughly

### Q: How should I structure my own robotics projects?
A: Organize projects following the ROS 2 package structure:
- Separate packages by functionality (sensors, control, perception)
- Use descriptive names and maintain consistent naming conventions
- Include proper package.xml and CMakeLists.txt files
- Document dependencies and build instructions

## Course Completion and Next Steps

### Q: What should I do after completing this textbook?
A: Consider these next steps:
- Participate in robotics competitions or hackathons
- Contribute to open-source robotics projects
- Explore research opportunities in physical AI
- Apply learned concepts to personal robotics projects
- Pursue advanced topics like reinforcement learning for robotics

### Q: Are there additional resources for deeper learning?
A: Yes, each chapter includes links to official documentation and research papers. The textbook also includes a comprehensive reference section in the appendices.