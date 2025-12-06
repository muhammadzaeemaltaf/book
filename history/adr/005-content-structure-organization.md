---
title: Content Structure - Hierarchical Modules with Learning Paths
status: Accepted
date: 2025-12-06
deciders: [Muhammad Zaeem Altaf]
consulted: []
informed: []
---

# ADR-005: Content Structure - Hierarchical Modules with Learning Paths

## Context

The Physical AI & Humanoid Robotics textbook must organize content in a way that supports progressive learning, with foundational concepts introduced before advanced implementations. The structure needs to accommodate 14 chapters across 4 modules while maintaining clear learning objectives, prerequisites, and cross-references. The organization should support both linear and non-linear learning paths for different student needs.

## Decision

We will organize content in hierarchical modules with clear learning paths following this structure:

- **Textbook Level**: Single comprehensive resource
- **Module Level**: 4 major learning units (ROS 2, Simulation, Isaac, VLA)
- **Chapter Level**: Focused learning units with specific objectives
- **Section Level**: Semantic content sections (introduction, concepts, tutorials, etc.)
- **Component Level**: Code examples, diagrams, assessments, and resources

The progression will follow: Introduction → Foundation (ROS 2) → Simulation → AI Integration → Capstone

## Alternatives Considered

1. **Flat Structure** - Would lack clear progression and prerequisite relationships
2. **Topic-based Organization** - Would make it harder to follow progressive learning
3. **Chronological Order** - Would not group related concepts effectively
4. **Skill-based Grouping** - Would be less intuitive for beginners
5. **Project-based Structure** - Would make it harder to isolate specific concepts

## Consequences

### Positive
- Progressive learning approach matches pedagogical requirements
- Clear prerequisite dependencies between chapters
- Supports both linear and non-linear learning paths
- Facilitates cross-references between related concepts
- Enables module-level mini-projects to validate learning
- Supports cross-module integration challenges
- Clear navigation and content discovery

### Negative
- Requires careful planning to maintain prerequisite dependencies
- May feel rigid for students who prefer non-linear learning
- Requires significant upfront design to maintain logical flow
- Cross-module references may be harder to discover
- Students may feel compelled to follow linear path when alternatives exist

## References

- plan.md: Project Structure section
- research.md: Architecture Patterns - Content Organization Pattern
- data-model.md: Content Hierarchy and Relationships
- spec.md: User stories and functional requirements