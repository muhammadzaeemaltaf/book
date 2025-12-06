---
title: Target Environment - Ubuntu 22.04 LTS with ROS 2 Humble
status: Accepted
date: 2025-12-06
deciders: [Muhammad Zaeem Altaf]
consulted: []
informed: []
---

# ADR-002: Target Environment - Ubuntu 22.04 LTS with ROS 2 Humble

## Context

The Physical AI & Humanoid Robotics textbook includes hands-on code examples that students must execute in their own environments. The choice of target operating system and robotics framework significantly impacts the accessibility, stability, and long-term maintainability of the educational content. The solution must balance cutting-edge capabilities with long-term support and widespread community adoption.

## Decision

We will target Ubuntu 22.04 LTS with ROS 2 Humble Hawksbill as the primary development environment. This decision includes:

- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Robotics Framework**: ROS 2 Humble Hawksbill (LTS version)
- **Python Version**: Python 3.10+ for code examples
- **Testing Environment**: Clean Ubuntu 22.04 + ROS 2 Humble VMs for validation
- **Version Locking**: Specific versions for reproducibility and stability

## Alternatives Considered

1. **ROS 2 Rolling** - Less stable, not suitable for textbook content requiring long-term stability
2. **Older ROS 2 versions** - Would miss newer features and improvements, shorter support window
3. **Different Linux distributions** - Less compatibility with robotics tools and community support
4. **Windows/WSL** - Additional complexity for students, less native ROS support
5. **Docker-based approach** - Adds complexity for beginners, requires additional learning

## Consequences

### Positive
- ROS 2 Humble is the LTS version with 5-year support (2022-2027)
- Ubuntu 22.04 is the officially supported OS for ROS 2 Humble
- Long-term stability for students learning the material
- Extensive official documentation and community support
- Reduces environment-specific errors and troubleshooting
- Ensures code examples remain functional over time

### Negative
- Students must use a specific OS/environment combination
- May exclude users on other platforms (Windows, macOS) without additional setup
- Requires students to potentially dual-boot or use VMs
- Limits flexibility for using newer ROS 2 features

## References

- plan.md: Technical Context section on dependencies
- research.md: Technology Stack Decisions - Code Example Environment
- data-model.md: Technical Constraints - Environment compatibility
- quickstart.md: System requirements and setup instructions