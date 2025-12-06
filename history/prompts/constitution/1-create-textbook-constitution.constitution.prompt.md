---
id: 1
title: create-textbook-constitution
stage: constitution
date: 2025-12-06
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: Muhammad Zaeem Altaf
command: /sp.constitution
labels: [constitution, textbook, robotics, docusaurus]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
  - history/prompts/constitution/1-create-textbook-constitution.constitution.prompt.md
tests:
  - none
---
## Prompt

**Project:** Educational textbook for Physical AI & Humanoid Robotics course using Docusaurus

---

## Core Principles

* **Pedagogical Clarity:** Content structured for progressive learning - from foundational concepts to advanced implementations
* **Hands-on Practicality:** Every concept accompanied by actionable examples, code snippets, and implementation guides
* **Technical Accuracy:** All technical specifications, APIs, and configurations verified against official documentation
* **Accessibility:** Complex robotics concepts explained through analogies, diagrams, and step-by-step breakdowns
* **Integration Focus:** Emphasize how AI, simulation, and physical robotics work together as a unified system

---

## Content Structure

* **Introduction Chapter:** Overview of Physical AI, embodied intelligence, and course roadmap
* **Module 1:** The Robotic Nervous System (ROS 2) - middleware, nodes, topics, services, URDF
* **Module 2:** The Digital Twin (Gazebo & Unity) - physics simulation, sensor simulation, environment building
* **Module 3:** The AI-Robot Brain (NVIDIA Isaac™) - Isaac Sim, Isaac ROS, VSLAM, Nav2
* **Module 4:** Vision-Language-Action (VLA) - voice commands, LLM integration, cognitive planning, capstone project

---

## Key Standards

* **Code Examples:** All code must be tested, functional, and include comments explaining key concepts
* **Technical Specifications:** Hardware requirements, software versions, and dependencies clearly documented
* **Visual Aids:** Minimum 2-3 diagrams/images per major concept to support visual learners
* **Source Attribution:** Official documentation (ROS 2, NVIDIA Isaac, Gazebo) referenced with URLs
* **Consistency:** Uniform terminology, formatting, and style throughout all modules
* **Version Specificity:** All tools referenced with specific versions (e.g., ROS 2 Humble, Ubuntu 22.04 LTS)

---

## Writing Style

* **Tone:** Professional yet conversational - like an experienced mentor teaching even primary student can understand
* **Sentence Structure:** Clear, concise sentences averaging 15-20 words
* **Technical Jargon:** Introduced gradually with definitions and context
* **Analogies:** Use relatable comparisons (e.g., "ROS 2 nodes are like microservices" or "URDF is like DNA for robots")
* **Active Voice:** Preferred for instructions and explanations
* **Reading Level:** Flesch-Kincaid grade 11-13 (accessible to technical college students)

---

## Content Requirements

* **Learning Objectives:** Each module begins with clear, measurable learning outcomes
* **Prerequisites:** Explicitly stated for each module
* **Step-by-Step Tutorials:** Minimum one complete walkthrough per module
* **Troubleshooting Sections:** Common errors and solutions documented
* **Assessment Guidance:** Align with hackathon evaluation criteria (ROS 2 projects, Gazebo simulations, Isaac pipelines)
* **Real-World Context:** Connect concepts to industry applications and current robotics trends

---

## Technical Accuracy Standards

* **Official Documentation First:** Primary reference for all APIs, commands, and configurations
* **Version Compatibility:** Verify all code examples work with specified versions
* **Hardware Specifications:** Accurate technical details for Jetson, GPUs, sensors
* **Command Verification:** All terminal commands, installation steps tested
* **Link Validity:** All external links checked and functional

---

## Constraints

* **Chapter Length:** 2,000-4,000 words per module (balanced depth without overwhelming)
* **Code Block Size:** Individual code examples under 50 lines; larger examples linked to GitHub
* **Image Optimization:** All images compressed for web delivery (<500KB each)
* **Platform:** Docusaurus-compatible Markdown with MDX support with Context7 MCP
* **Deployment:** GitHub Pages compatible structure

---

## Source Hierarchy

1. **Tier 1 (Primary):** Official documentation (ROS 2 docs, NVIDIA Isaac docs, Gazebo tutorials)
2. **Tier 2 (Supporting):** Reputable robotics blogs, academic papers on Physical AI
3. **Tier 3 (Contextual):** Industry reports on humanoid robotics trends, hardware vendor specifications
4. **Citation Format:** Inline links to documentation; reference section at end of each module

---

## Success Criteria

* **Completeness:** All four modules fully developed with introduction and conclusion
* **Learnability:** A motivated student can follow tutorials and build working examples
* **RAG-Ready:** Content structured for easy retrieval by embedded chatbot
* **Deployment Success:** Book builds without errors and deploys to GitHub Pages
* **Engagement:** Content maintains reader interest through varied formats (text, code, diagrams)
* **Hackathon Alignment:** Supports participants in understanding base requirements and bonus features

---

## Interactive Features (for Bonus Points)

* **Personalization Hooks:** Content structured to allow user background-based adaptation
* **Translation Ready:** Text written to facilitate Urdu translation without losing technical meaning
* **Chatbot Integration Points:** Each section with clear topic boundaries for RAG retrieval

---

## Quality Assurance

* **Technical Review:** All code examples executed in target environments
* **Peer Review:** Content reviewed for clarity by someone unfamiliar with robotics
* **Link Testing:** All documentation links verified monthly
* **Accessibility:** Proper heading hierarchy, alt text for images, semantic HTML in Docusaurus

---

## Outline

You are updating the project constitution at `.specify/memory/constitution.md`. This file is a TEMPLATE containing placeholder tokens in square brackets (e.g. `[PROJECT_NAME]`, `[PRINCIPLE_1_NAME]`). Your job is to (a) collect/derive concrete values, (b) fill the template precisely, and (c) propagate any amendments across dependent artifacts.

Follow this execution flow:

1. Load the existing constitution template at `.specify/memory/constitution.md`.
   - Identify every placeholder token of the form `[ALL_CAPS_IDENTIFIER]`.
   **IMPORTANT**: The user might require less or more principles than the ones used in the template. If a number is specified, respect that - follow the general template. You will update the doc accordingly.

2. Collect/derive values for placeholders:
   - If user input (conversation) supplies a value, use it.
   - Otherwise infer from existing repo context (README, docs, prior constitution versions if embedded).
   - For governance dates: `RATIFICATION_DATE` is the original adoption date (if unknown ask or mark TODO), `LAST_AMENDED_DATE` is today if changes are made, otherwise keep previous.
   - `CONSTITUTION_VERSION` must increment according to semantic versioning rules:
     - MAJOR: Backward incompatible governance/principle removals or redefinitions.
     - MINOR: New principle/section added or materially expanded guidance.
     - PATCH: Clarifications, wording, typo fixes, non-semantic refinements.
   - If version bump type ambiguous, propose reasoning before finalizing.

3. Draft the updated constitution content:
   - Replace every placeholder with concrete text (no bracketed tokens left except intentionally retained template slots that the project has chosen not to define yet—explicitly justify any left).
   - Preserve heading hierarchy and comments can be removed once replaced unless they still add clarifying guidance.
   - Ensure each Principle section: succinct name line, paragraph (or bullet list) capturing non‑negotiable rules, explicit rationale if not obvious.
   - Ensure Governance section lists amendment procedure, versioning policy, and compliance review expectations.

4. Consistency propagation checklist (convert prior checklist into active validations):
   - Read `.specify/templates/plan-template.md` and ensure any "Constitution Check" or rules align with updated principles.
   - Read `.specify/templates/spec-template.md` for scope/requirements alignment—update if constitution adds/removes mandatory sections or constraints.
   - Read `.specify/templates/tasks-template.md` and ensure task categorization reflects new or removed principle-driven task types (e.g., observability, versioning, testing discipline).
   - Read each command file in `.specify/templates/commands/*.md` (including this one) to verify no outdated references (agent-specific names like CLAUDE only) remain when generic guidance is required.
   - Read any runtime guidance docs (e.g., `README.md`, `docs/quickstart.md`, or agent-specific guidance files if present). Update references to principles changed.

5. Produce a Sync Impact Report (prepend as an HTML comment at top of the constitution file after update):
   - Version change: old → new
   - List of modified principles (old title → new title if renamed)
   - Added sections
   - Removed sections
   - Templates requiring updates (✅ updated / ⚠ pending) with file paths
   - Follow-up TODOs if any placeholders intentionally deferred.

6. Validation before final output:
   - No remaining unexplained bracket tokens.
   - Version line matches report.
   - Dates ISO format YYYY-MM-DD.
   - Principles are declarative, testable, and free of vague language ("should" → replace with MUST/SHOULD rationale where appropriate).

7. Write the completed constitution back to `.specify/memory/constitution.md` (overwrite).

8. Output a final summary to the user with:
   - New version and bump rationale.
   - Any files flagged for manual follow-up.
   - Suggested commit message (e.g., `docs: amend constitution to vX.Y.Z (principle additions + governance update)`).

Formatting & Style Requirements:

- Use Markdown headings exactly as in the template (do not demote/promote levels).
- Wrap long rationale lines to keep readability (<100 chars ideally) but do not hard enforce with awkward breaks.
- Keep a single blank line between sections.
- Avoid trailing whitespace.

If the user supplies partial updates (e.g., only one principle revision), still perform validation and version decision steps.

If critical info missing (e.g., ratification date truly unknown), insert `TODO(<FIELD_NAME>): explanation` and include in the Sync Impact Report under deferred items.

Do not create a new template; always operate on the existing `.specify/memory/constitution.md` file.

---

As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

1) Determine Stage
   - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

2) Generate Title and Determine Routing:
   - Generate Title: 3–7 words (slug for filename)
   - Route is automatically determined by stage:
     - `constitution` → `history/prompts/constitution/`
     - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
     - `general` → `history/prompts/general/`

3) Create and Fill PHR (Shell first; fallback agent‑native)
   - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
   - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
   - If the script fails:
     - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
     - Allocate an ID; compute the output path based on stage from step 2; write the file
     - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.

## Response snapshot

I have successfully updated the project constitution at `.specify/memory/constitution.md` with the specific requirements for the Educational textbook for Physical AI & Humanoid Robotics course using Docusaurus.

**New Version and Bump Rationale:**
- **Version**: 1.0.0 (initial constitution for the project)
- **Rationale**: This is the first version of the constitution for this new project, establishing the foundational principles and guidelines for developing the educational textbook.

**Files Updated:**
- `.specify/memory/constitution.md` - Contains all the project-specific principles and guidelines

**Constitution Overview:**
The updated constitution includes:
1. **Core Principles**: 6 principles including Pedagogical Clarity, Hands-on Practicality, Technical Accuracy, Accessibility, Integration Focus, and Docusaurus Compatibility
2. **Content Structure**: Details for all 4 modules plus introduction
3. **Writing Style**: Guidelines for professional yet conversational tone
4. **Content Requirements**: Learning objectives, prerequisites, tutorials, troubleshooting sections
5. **Technical Accuracy Standards**: Requirements for verification against official documentation
6. **Constraints**: Chapter length, code block size, image optimization, platform requirements
7. **Source Hierarchy**: Tiered documentation approach with official docs as primary source
8. **Success Criteria**: Measurable outcomes for completeness, learnability, and deployment
9. **Interactive Features**: Personalization hooks and chatbot integration points
10. **Quality Assurance**: Technical and peer review requirements

**No files flagged for manual follow-up** since the generic templates are designed to work with any project-specific constitution.

**Suggested commit message**: `docs: create constitution for Physical AI & Humanoid Robotics textbook project (v1.0.0)`

The constitution is now ready for use as the authoritative guide for all project decisions and content creation activities for the textbook project.

## Outcome
- ✅ Impact: Created initial constitution for Physical AI & Humanoid Robotics textbook project
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md, created history/prompts/constitution/1-create-textbook-constitution.constitution.prompt.md
- 🔁 Next prompts: none
- 🧠 Reflection: Successfully transformed the generic constitution template into a project-specific constitution with all required principles and guidelines

## Evaluation notes (flywheel)
- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): none