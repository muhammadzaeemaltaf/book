# Data Model: Physical AI & Humanoid Robotics Textbook

## Overview
This document defines the conceptual data model for the Physical AI & Humanoid Robotics textbook content management. Since this is a documentation project, the "data model" represents the content structure and relationships rather than traditional database entities.

## Core Entities

### 1. Textbook Module
**Description**: A major learning unit containing multiple chapters with related concepts
**Attributes**:
- moduleId (string): Unique identifier (e.g., "module-01-ros2")
- title (string): Display title of the module
- description (string): Brief overview of module content
- learningObjectives (array of strings): Measurable learning outcomes
- prerequisites (array of strings): Required knowledge/skills before starting
- estimatedDuration (number): Time to complete in hours
- chapterIds (array of strings): Ordered list of contained chapter IDs

**Relationships**:
- Contains 1..n Chapters
- May depend on other Modules (prerequisite relationship)

### 2. Chapter
**Description**: A focused learning unit within a module with specific learning objectives
**Attributes**:
- chapterId (string): Unique identifier (e.g., "chapter-01-01-architecture")
- moduleId (string): Parent module identifier
- title (string): Display title of the chapter
- description (string): Brief overview of chapter content
- learningObjectives (array of strings): Measurable learning outcomes
- prerequisites (array of strings): Required knowledge/skills before starting
- wordCount (number): Length of chapter content
- estimatedDuration (number): Time to complete in minutes
- contentStructure (object): Hierarchical organization of content sections

**Relationships**:
- Belongs to 1 Module
- Contains 1..n Content Sections
- Contains 0..n Code Examples
- Contains 0..n Diagrams

### 3. Content Section
**Description**: A semantic section within a chapter (e.g., introduction, core concepts, tutorial)
**Attributes**:
- sectionId (string): Unique identifier
- chapterId (string): Parent chapter identifier
- title (string): Section heading
- sectionType (string): Type of content (introduction, concept, tutorial, troubleshooting, etc.)
- content (string): Markdown content of the section
- orderIndex (number): Position within the chapter

**Relationships**:
- Belongs to 1 Chapter
- May reference 0..n Other Content Sections (cross-references)
- May contain 0..n Code Examples
- May contain 0..n Diagrams

### 4. Code Example
**Description**: A runnable code snippet with associated metadata
**Attributes**:
- exampleId (string): Unique identifier
- chapterId (string): Parent chapter identifier
- sectionId (string): Parent section identifier (optional)
- title (string): Descriptive title for the example
- language (string): Programming language (python, bash, xml, etc.)
- code (string): The actual code content
- description (string): Explanation of what the code does
- expectedOutput (string): Expected results when code is run
- prerequisites (array of strings): Setup required before running
- difficultyLevel (string): beginner, intermediate, or advanced
- testedEnvironment (string): Environment where code was verified
- fileLocation (string): Path for downloadable version (if applicable)

**Relationships**:
- Belongs to 1 Chapter
- Belongs to 1 Content Section (optional)
- May reference 0..n Other Code Examples (dependencies)

### 5. Diagram
**Description**: Visual aid to support understanding of concepts
**Attributes**:
- diagramId (string): Unique identifier
- chapterId (string): Parent chapter identifier
- sectionId (string): Parent section identifier (optional)
- title (string): Descriptive title for the diagram
- description (string): Explanation of what the diagram illustrates
- imageUrl (string): Path to the image file
- altText (string): Accessibility text for screen readers
- type (string): Conceptual, architectural, process, etc.
- fileLocation (string): Path to the image file in static/img/

**Relationships**:
- Belongs to 1 Chapter
- Belongs to 1 Content Section (optional)
- May be referenced by 0..n Content Sections

### 6. Troubleshooting Guide
**Description**: Collection of common errors and their solutions
**Attributes**:
- guideId (string): Unique identifier
- chapterId (string): Parent chapter identifier
- sectionId (string): Parent section identifier (optional)
- title (string): Descriptive title
- errorEntries (array of objects): List of specific error/solution pairs
  - errorDescription (string): What the error looks like
  - cause (string): Why the error occurs
  - solution (string): Step-by-step fix
  - preventionTips (array of strings): How to avoid the error

**Relationships**:
- Belongs to 1 Chapter
- Belongs to 1 Content Section (optional)
- May reference 0..n Code Examples (for error reproduction)

### 7. Learning Assessment
**Description**: Self-assessment questions and challenges for a chapter
**Attributes**:
- assessmentId (string): Unique identifier
- chapterId (string): Parent chapter identifier
- title (string): Descriptive title
- knowledgeQuestions (array of objects): Factual understanding questions
  - question (string): The question text
  - options (array of strings): Multiple choice options (if applicable)
  - correctAnswer (string): The correct response
  - explanation (string): Why this is correct
- practicalChallenges (array of objects): Hands-on challenges
  - challenge (string): The practical task
  - difficultyLevel (string): beginner, intermediate, advanced
  - expectedOutcome (string): What successful completion looks like
  - hints (array of strings): Guidance for students

**Relationships**:
- Belongs to 1 Chapter
- References 1..n Content Sections (tested concepts)

### 8. External Resource
**Description**: Links to official documentation and supplementary materials
**Attributes**:
- resourceId (string): Unique identifier
- chapterId (string): Parent chapter identifier (optional, null for general resources)
- sectionId (string): Parent section identifier (optional)
- title (string): Descriptive title of the resource
- url (string): Full URL to the resource
- resourceType (string): Documentation, tutorial, video, paper, etc.
- verificationDate (string): When the link was last checked
- status (string): active, broken, redirected
- relevance (string): How the resource relates to chapter content

**Relationships**:
- Belongs to 1 Chapter (optional, for general resources)
- Belongs to 1 Content Section (optional)

## Content Relationships

### Module Progression Dependencies
```
Introduction → Module 1 (ROS 2) → Module 2 (Simulation) → Module 3 (Isaac) → Module 4 (VLA) → Capstone
```

### Cross-Module References
- Module 2 references Module 1 concepts (ROS 2 integration with simulation)
- Module 3 references Module 1 and 2 concepts (Isaac ROS builds on ROS 2 and simulation)
- Module 4 references all previous modules (VLA integration)

### Content Hierarchy
```
Textbook
├── Module
    ├── Chapter
        ├── Content Section
        │   ├── Code Example
        │   ├── Diagram
        │   ├── Troubleshooting Guide
        │   └── External Resource
        ├── Learning Assessment
        └── External Resources
```

## Validation Rules

### 1. Module Validation
- Each module must have 1..n chapters
- Learning objectives must be measurable and specific
- Prerequisites must be satisfied by previous content or be foundational knowledge
- Estimated duration must be between 2-8 hours per module

### 2. Chapter Validation
- Each chapter must have 1 learning objective section
- Each chapter must have 1 prerequisites section
- Each chapter must have 1 introduction section
- Each chapter must have 1 hands-on tutorial section
- Each chapter must have 1 troubleshooting section
- Each chapter must have 1 key takeaways section
- Each chapter must have 1 self-assessment section
- Word count must be between 2,000-4,000 words
- Estimated duration must be between 30-90 minutes

### 3. Code Example Validation
- All code must be tested and functional in specified environment
- Code blocks must be less than 50 lines (split if longer)
- All dependencies must be clearly listed
- Expected output must be documented
- Tested environment must match project requirements (Ubuntu 22.04 + ROS 2 Humble)

### 4. Diagram Validation
- All images must be optimized (<500KB)
- Alt text must be provided for accessibility
- Diagrams must support content understanding
- Minimum 2 diagrams per major concept as specified in requirements

### 5. Assessment Validation
- Knowledge questions must align with learning objectives
- Practical challenges must be achievable with provided content
- Difficulty levels must match chapter complexity
- Assessments must provide meaningful feedback to students

## State Transitions

### Content Development States
1. **Draft**: Initial content creation
2. **Reviewed**: Peer review completed
3. **Validated**: Technical accuracy verified
4. **Published**: Available in textbook

### Content Relationships
- Each entity has a lifecycle state attribute
- State transitions are tracked with timestamps and responsible party
- Validation gates exist between states to ensure quality

## Constraints

### Structural Constraints
- Maximum 4 modules in the textbook
- Maximum 4 chapters per module (except Module 4 with 4 chapters)
- Maximum 50 code examples per chapter
- Maximum 30 diagrams per chapter
- Content must be Docusaurus-compatible Markdown

### Quality Constraints
- All content must adhere to pedagogical clarity principles
- All code examples must be technically accurate
- All external links must be verified and maintained
- Content must be accessible to students with Python/AI background
- Reading level must be Flesch-Kincaid grade 11-13

### Technical Constraints
- Compatible with Docusaurus 3.x
- Deployable to GitHub Pages
- Integratable with RAG chatbot system
- Supportable for Context7 MCP integration
- Optimized for fast page loading (<3 seconds)