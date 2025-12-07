---
id: troubleshooting-guide
title: "Troubleshooting Guide"
sidebar_label: "Troubleshooting Guide"
sidebar_position: 1
description: "Common errors and solutions for Physical AI & Humanoid Robotics textbook exercises"
keywords: [troubleshooting, errors, solutions, debugging, robotics]
---

# Troubleshooting Guide

This guide provides solutions to common issues encountered when working with the Physical AI & Humanoid Robotics textbook examples.

## ROS 2 Common Issues

### 1. Node Communication Problems
**Symptom**: Nodes cannot communicate with each other
**Cause**: Different DDS domains or network configuration issues
**Solution**:
- Verify both nodes are on the same network
- Check that ROS_DOMAIN_ID is set consistently across nodes
- Ensure firewall isn't blocking DDS communication

### 2. Package Build Errors
**Symptom**: `colcon build` fails with compilation errors
**Cause**: Missing dependencies or incorrect CMakeLists.txt configuration
**Solution**:
- Verify all dependencies are declared in package.xml
- Check CMakeLists.txt for proper find_package() calls
- Ensure workspace is sourced after dependency installation

### 3. Import Errors After Build
**Symptom**: Python modules not found after successful build
**Cause**: Workspace not sourced after building
**Solution**: Run `source install/setup.bash` in your workspace directory

## Isaac Sim Issues

### 1. Simulation Not Starting
**Symptom**: Isaac Sim fails to launch or crashes immediately
**Cause**: GPU driver or CUDA compatibility issues
**Solution**:
- Verify NVIDIA GPU with latest drivers
- Check CUDA compatibility with Isaac Sim version
- Ensure sufficient VRAM for simulation

### 2. Sensor Data Not Publishing
**Symptom**: Camera, LiDAR, or other sensors not publishing data
**Cause**: Sensor configuration or rendering pipeline issues
**Solution**:
- Verify sensor is properly attached to a prim
- Check rendering mode is set to "rtx-render" or appropriate mode
- Ensure Isaac Sim extension is enabled

## VLA Integration Issues

### 1. Model Loading Failures
**Symptom**: VLA models fail to load or throw memory errors
**Cause**: Insufficient GPU memory or model format issues
**Solution**:
- Check GPU memory availability
- Verify model file format compatibility
- Try loading model on CPU if GPU memory is insufficient

### 2. Inference Performance Problems
**Symptom**: Slow inference times or dropped frames
**Cause**: Computational overload or inefficient model implementation
**Solution**:
- Optimize batch sizes for your hardware
- Consider model quantization for faster inference
- Reduce input resolution if acceptable for your application

## General Development Environment

### 1. Permission Issues
**Symptom**: Permission denied errors when running scripts
**Cause**: Insufficient file permissions
**Solution**: Run `chmod +x script_name.py` to make scripts executable

### 2. Dependency Conflicts
**Symptom**: Import errors related to conflicting package versions
**Cause**: Multiple Python environments or conflicting dependencies
**Solution**:
- Use virtual environments for project isolation
- Create a dedicated conda environment for robotics development
- Verify package versions match textbook requirements

## Quick Reference Commands

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Source your workspace after building
source install/setup.bash

# Check active ROS 2 nodes
ros2 node list

# Check active ROS 2 topics
ros2 topic list

# Check available ROS 2 services
ros2 service list

# View logs for a specific node
ros2 launch-logs --node-name node_name
```

## Getting Additional Help

If the solutions above don't resolve your issue:

1. Check the [ROS 2 documentation](https://docs.ros.org/en/humble/) for detailed troubleshooting
2. Review the [NVIDIA Isaac documentation](https://docs.nvidia.com/isaac/) for Isaac Sim specific issues
3. Search the [Gazebo community forums](https://community.gazebosim.org/) for simulation issues
4. Verify your system meets the hardware requirements in the textbook's appendix

## System Requirements Verification

```bash
# Check ROS 2 installation
ros2 --version

# Check Python version
python3 --version

# Check available GPU memory (NVIDIA)
nvidia-smi

# Check available system memory
free -h

# Check disk space
df -h
```