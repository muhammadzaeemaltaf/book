---
title: Simulation Environment Strategy - Gazebo and Isaac Sim Support
status: Accepted
date: 2025-12-06
deciders: [Muhammad Zaeem Altaf]
consulted: []
informed: []
---

# ADR-003: Simulation Environment Strategy - Gazebo and Isaac Sim Support

## Context

The Physical AI & Humanoid Robotics textbook covers simulation concepts that require practical implementation. Students have varying access to hardware, particularly high-end GPUs required for advanced simulation. The solution must balance accessibility for students with limited hardware resources while providing access to state-of-the-art simulation capabilities for those with appropriate hardware.

## Decision

We will support both Gazebo Garden and NVIDIA Isaac Sim with clear guidance on when to use each. This decision includes:

- **Primary Tutorials**: Use Gazebo for accessibility and compatibility with standard hardware
- **Advanced Content**: Cover Isaac Sim for state-of-the-art Physical AI development
- **Hardware Guidance**: Clear documentation on hardware requirements vs. cloud alternatives
- **Cloud Options**: Include guidance for using Isaac Sim via cloud services (AWS G5 instances)
- **Decision Matrix**: Provide clear criteria for choosing between simulation environments

## Alternatives Considered

1. **Gazebo Only** - Would limit access to cutting-edge simulation capabilities
2. **Isaac Sim Only** - Would exclude students without RTX-class GPUs
3. **Unity Only** - Less integration with ROS 2 ecosystem
4. **Custom Simulation Framework** - Would require significant development and maintenance effort

## Consequences

### Positive
- Gazebo: Open-source, accessible to students without high-end GPUs
- Isaac Sim: State-of-the-art for Physical AI research and development
- Provides both accessible and cutting-edge options
- Cloud alternatives available for Isaac Sim for students without RTX hardware
- Students can progress from basic to advanced simulation concepts
- Aligns with industry practices using multiple simulation tools

### Negative
- Requires maintaining documentation for multiple simulation environments
- Increases complexity of code examples and troubleshooting
- May confuse beginners about which environment to choose
- Requires testing and validation across multiple platforms

## References

- plan.md: Technical Context section on dependencies
- research.md: Technology Stack Decisions - Simulation Environment Integration
- data-model.md: Content relationships and cross-module references
- quickstart.md: Hardware requirements and setup guidance