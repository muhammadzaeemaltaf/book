---
id: software-setup
title: "Software Setup Guide"
sidebar_label: "Software Setup"
sidebar_position: 4
description: "Detailed installation guides for Physical AI & Humanoid Robotics development environment"
keywords: [software, setup, installation, ros2, isaac, ubuntu, dependencies]
---

# Software Setup Guide

This guide provides detailed instructions for setting up your development environment for the Physical AI & Humanoid Robotics textbook.

## Prerequisites

### System Requirements
- Ubuntu 22.04 LTS (fresh installation recommended)
- Internet connection for package downloads
- Administrative (sudo) access

### Initial System Setup
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install basic development tools
sudo apt install build-essential cmake git python3-pip python3-dev python3-venv -y

# Install essential utilities
sudo apt install curl wget vim htop tmux screen -y
```

## ROS 2 Humble Hawksbill Installation

### 1. Set Locale
```bash
locale  # Check locale settings
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### 2. Add ROS 2 Repository
```bash
# Add the ROS 2 GPG key
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add the repository to your sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 3. Install ROS 2
```bash
sudo apt update
sudo apt install ros-humble-desktop-full -y

# Install colcon build tool
sudo apt install python3-colcon-common-extensions -y

# Install ROS 2 dependencies
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y

# Initialize rosdep
sudo rosdep init
rosdep update
```

### 4. Environment Setup
```bash
# Add ROS 2 to your bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## NVIDIA Isaac Sim Installation

### 1. Install NVIDIA GPU Drivers
```bash
# Check if NVIDIA GPU is detected
lspci | grep -i nvidia

# Install latest NVIDIA drivers
sudo apt install nvidia-driver-535 -y

# Reboot to apply changes
sudo reboot
```

### 2. Install Isaac Sim Dependencies
```bash
# Install CUDA (if not installed with driver)
sudo apt install nvidia-cuda-toolkit -y

# Install additional dependencies
sudo apt install python3-opencv python3-scipy python3-matplotlib -y
```

### 3. Verify Installation
```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version
```

## Python Environment Setup

### 1. Create Virtual Environment
```bash
# Create a dedicated environment for robotics development
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install common Python packages
pip install numpy scipy matplotlib pandas jupyter notebook torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Additional Python Dependencies
```bash
# Activate environment
source ~/robotics_env/bin/activate

# Install robotics-specific packages
pip install transforms3d pyquaternion opencv-python

# Install AI/ML packages
pip install transformers datasets accelerate
```

## Development Tools Installation

### 1. Version Control
```bash
# Install Git and configure
sudo apt install git -y
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global core.editor "vim"
```

### 2. Code Editor
```bash
# Option 1: Install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code -y

# Option 2: Install vim with plugins
sudo apt install vim-gtk3 -y
```

### 3. Network Tools
```bash
# Install network utilities
sudo apt install net-tools iputils-ping dnsutils -y

# Install SSH server (if needed)
sudo apt install openssh-server -y
```

## Verification and Testing

### 1. Test ROS 2 Installation
```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Check ROS 2 version
ros2 --version

# Run a simple test
ros2 run demo_nodes_cpp talker
```

### 2. Test Python Environment
```bash
# Activate environment
source ~/robotics_env/bin/activate

# Test Python packages
python3 -c "import numpy, scipy, matplotlib; print('Python packages OK')"

# Test PyTorch with CUDA
python3 -c "import torch; print(f'PyTorch OK, CUDA available: {torch.cuda.is_available()}')"
```

### 3. Create Test Workspace
```bash
# Create a test workspace
mkdir -p ~/test_workspace/src
cd ~/test_workspace

# Build empty workspace to verify setup
source /opt/ros/humble/setup.bash
colcon build

# Source the workspace
source install/setup.bash
```

## Common Installation Issues and Solutions

### 1. Permission Issues
**Problem**: Permission denied when installing packages
**Solution**: Use `sudo` for system packages, or check user group membership
```bash
# Check if user is in dialout group (needed for ROS serial communication)
groups $USER
# If not, add to group: sudo usermod -a -G dialout $USER
```

### 2. Network Issues
**Problem**: apt update fails due to network issues
**Solution**: Check network configuration and proxy settings
```bash
# Test network connectivity
ping 8.8.8.8
```

### 3. Disk Space Issues
**Problem**: Installation fails due to insufficient disk space
**Solution**: Check available space and free up if needed
```bash
# Check disk usage
df -h
# Check specific directories
du -sh /tmp /var/log
```

### 4. Repository Issues
**Problem**: ROS 2 repository not found
**Solution**: Verify Ubuntu version and ROS 2 distribution compatibility
```bash
# Check Ubuntu version
lsb_release -a
# ROS 2 Humble requires Ubuntu 22.04
```

## Environment Configuration

### 1. Bash Profile Setup
Add these lines to `~/.bashrc` for convenience:
```bash
# ROS 2 setup
source /opt/ros/humble/setup.bash

# Robotics environment
source ~/robotics_env/bin/activate

# Custom aliases
alias cw='cd ~/ros2_workspace'
alias cs='cd ~/ros2_workspace/src'
alias cb='cd ~/ros2_workspace && colcon build && source install/setup.bash'
```

### 2. Apply Changes
```bash
source ~/.bashrc
```

## Troubleshooting Checklist

Before starting the textbook exercises, verify:

- [ ] Ubuntu 22.04 LTS is installed
- [ ] NVIDIA GPU drivers are working (`nvidia-smi`)
- [ ] ROS 2 Humble is installed and sourced
- [ ] Python virtual environment is set up
- [ ] Basic ROS 2 commands work (`ros2 --version`)
- [ ] Network connectivity is available
- [ ] Sufficient disk space is available (100GB+ recommended)

## Additional Resources

- [ROS 2 Installation Guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- [NVIDIA Isaac Sim Installation](https://docs.nvidia.com/isaac/isaac_sim/install_basic.html)
- [Ubuntu 22.04 Setup Guide](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview)