<!--
Sync Impact Report:
- Version change: N/A → 1.0.0 (Initial constitution for the Physical AI & Humanoid Robotics textbook project)
- Added sections: Core Principles (6 principles), Content Structure, Writing Style, Content Requirements, Technical Accuracy Standards, Constraints, Source Hierarchy, Success Criteria, Interactive Features, Quality Assurance
- Removed sections: None (new constitution)
- Templates requiring updates: N/A (new project)
- Follow-up TODOs: None
-->
# Educational textbook for Physical AI & Humanoid Robotics course Constitution

## Core Principles

### Pedagogical Clarity
Content structured for progressive learning - from foundational concepts to advanced implementations. All content must follow a logical sequence that builds understanding incrementally, with each concept properly introduced before advancing to more complex topics.

### Hands-on Practicality
Every concept accompanied by actionable examples, code snippets, and implementation guides. Theory must be immediately followed by practical application so students can verify their understanding through hands-on exercises.

### Technical Accuracy
All technical specifications, APIs, and configurations verified against official documentation. Every code example, command, and technical detail must be tested and confirmed to work with the specified versions of tools and platforms.

### Accessibility
Complex robotics concepts explained through analogies, diagrams, and step-by-step breakdowns. Content must be accessible to students with varying technical backgrounds, using relatable comparisons and visual aids to enhance comprehension.

### Integration Focus
Emphasize how AI, simulation, and physical robotics work together as a unified system. All modules must demonstrate the interconnections between different technologies rather than treating them as isolated components.

### Docusaurus Compatibility
Content must be structured for Docusaurus deployment with MDX support and Context7 MCP integration. All content must follow Docusaurus-compatible Markdown standards with proper formatting for web delivery.

## Content Structure

* **Introduction Chapter:** Overview of Physical AI, embodied intelligence, and course roadmap
* **Module 1:** The Robotic Nervous System (ROS 2) - middleware, nodes, topics, services, URDF
* **Module 2:** The Digital Twin (Gazebo & Unity) - physics simulation, sensor simulation, environment building
* **Module 3:** The AI-Robot Brain (NVIDIA Isaac™) - Isaac Sim, Isaac ROS, VSLAM, Nav2
* **Module 4:** Vision-Language-Action (VLA) - voice commands, LLM integration, cognitive planning, capstone project

## Writing Style

* **Tone:** Professional yet conversational - like an experienced mentor teaching even primary student can understand
* **Sentence Structure:** Clear, concise sentences averaging 15-20 words
* **Technical Jargon:** Introduced gradually with definitions and context
* **Analogies:** Use relatable comparisons (e.g., "ROS 2 nodes are like microservices" or "URDF is like DNA for robots")
* **Active Voice:** Preferred for instructions and explanations
* **Reading Level:** Flesch-Kincaid grade 11-13 (accessible to technical college students)

## Content Requirements

* **Learning Objectives:** Each module begins with clear, measurable learning outcomes
* **Prerequisites:** Explicitly stated for each module
* **Step-by-Step Tutorials:** Minimum one complete walkthrough per module
* **Troubleshooting Sections:** Common errors and solutions documented
* **Assessment Guidance:** Align with hackathon evaluation criteria (ROS 2 projects, Gazebo simulations, Isaac pipelines)
* **Real-World Context:** Connect concepts to industry applications and current robotics trends

## Technical Accuracy Standards

* **Official Documentation First:** Primary reference for all APIs, commands, and configurations
* **Version Compatibility:** Verify all code examples work with specified versions
* **Hardware Specifications:** Accurate technical details for Jetson, GPUs, sensors
* **Command Verification:** All terminal commands, installation steps tested
* **Link Validity:** All external links checked and functional

## Constraints

* **Chapter Length:** 2,000-4,000 words per module (balanced depth without overwhelming)
* **Code Block Size:** Individual code examples under 50 lines; larger examples linked to GitHub
* **Image Optimization:** All images compressed for web delivery (<500KB each)
* **Platform:** Docusaurus-compatible Markdown with MDX support with Context7 MCP
* **Deployment:** GitHub Pages compatible structure

## Source Hierarchy

1. **Tier 1 (Primary):** Official documentation (ROS 2 docs, NVIDIA Isaac docs, Gazebo tutorials)
2. **Tier 2 (Supporting):** Reputable robotics blogs, academic papers on Physical AI
3. **Tier 3 (Contextual):** Industry reports on humanoid robotics trends, hardware vendor specifications
4. **Citation Format:** Inline links to documentation; reference section at end of each module

## Success Criteria

* **Completeness:** All four modules fully developed with introduction and conclusion
* **Learnability:** A motivated student can follow tutorials and build working examples
* **RAG-Ready:** Content structured for easy retrieval by embedded chatbot
* **Deployment Success:** Book builds without errors and deploys to GitHub Pages
* **Engagement:** Content maintains reader interest through varied formats (text, code, diagrams)
* **Hackathon Alignment:** Supports participants in understanding base requirements and bonus features

## Interactive Features (for Bonus Points)

* **Personalization Hooks:** Content structured to allow user background-based adaptation
* **Translation Ready:** Text written to facilitate Urdu translation without losing technical meaning
* **Chatbot Integration Points:** Each section with clear topic boundaries for RAG retrieval

## Quality Assurance

* **Technical Review:** All code examples executed in target environments
* **Peer Review:** Content reviewed for clarity by someone unfamiliar with robotics
* **Link Testing:** All documentation links verified monthly
* **Accessibility:** Proper heading hierarchy, alt text for images, semantic HTML in Docusaurus

## Governance

All content development must adhere to the specified principles and standards. Any deviations from the established structure, writing style, or technical accuracy standards require explicit justification and approval. The constitution serves as the authoritative guide for all project decisions and content creation activities.

**Version**: 1.0.0 | **Ratified**: 2025-12-06 | **Last Amended**: 2025-12-06