---
id: 2
title: Multi agent implementation
stage: misc
date: 2025-12-10
surface: agent
model: claude-sonnet-4-5-20250929
feature: 002-rag-chatbot
branch: 002-rag-chatbot
user: muhammad-zaeem-altaf
command: manual
labels: [chatbot, agents, architecture, multi-agent]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - /home/muhammad-zaeem-altaf/Personal/project/book/backend/src/agents/greeting_agent.py
  - /home/muhammad-zaeem-altaf/Personal/project/book/backend/src/agents/book_agent.py
  - /home/muhammad-zaeem-altaf/Personal/project/book/backend/src/services/chat_service.py
tests:
---

## Prompt

The provided text outlines the learning objectives for a module from the 'Guide to Physical AI & Humanoid Robotics' book. By the end of this module, you will be able to: * Understand the fundamental architecture and core concepts of ROS 2. * Create and effectively manage ROS 2 packages and workspaces. * Implement nodes, topics, services, and actions, which are key components for robotic communication within ROS 2. * Describe robot structures using URDF (Unified Robot Description Format). * Build your first ROS 2 package, specifically incorporating publishers and subscribers for data exchange. > this is agent response fix it it should be show in proper format and alignment.

## Response snapshot

Implemented a sophisticated multi-agent architecture for the RAG chatbot:

1. **Greeting Agent** (`greeting_agent.py`): Handles greetings, farewells, and thank you messages with appropriate system prompts and detection logic.

2. **Book Agent** (`book_agent.py`): Handles book-related questions with RAG retrieval and intelligent fallback to general knowledge when no context is found. Includes topic detection for robotics/AI related queries.

3. **Enhanced Chat Service** (`chat_service.py`): Updated to route queries to appropriate agents based on content type. Uses a two-tier approach:
   - First, checks if query is a greeting using the greeting agent
   - If not, routes to the book agent for RAG retrieval and response generation

4. **Intelligent Fallback System**: The book agent can detect if queries are related to book topics and respond accordingly, either with retrieved context or general knowledge.

5. **Context-Aware Prompts**: Different system prompts are used based on whether context was found, the query type, and the chat mode (normal vs selected text).

## Outcome

- ✅ Impact: Created a more sophisticated and scalable agent-based architecture
- 🧪 Tests: Manual verification of agent routing functionality
- 📁 Files: Created greeting_agent.py, book_agent.py; updated chat_service.py
- 🔁 Next prompts: N/A
- 🧠 Reflection: Multi-agent approach provides better separation of concerns and extensibility

## Evaluation notes (flywheel)

- Failure modes observed: API rate limiting issues with Google Gemini API
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Implement API rate limiting handling