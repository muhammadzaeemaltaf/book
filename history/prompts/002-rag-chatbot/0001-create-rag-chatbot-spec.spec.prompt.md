---
id: 0001
title: Create RAG Chatbot Spec
stage: spec
date: 2025-12-09
surface: agent
model: Claude Sonnet 4.5
feature: 002-rag-chatbot
branch: 002-rag-chatbot
user: Muhammad Zaeem Altaf
command: /sp.specify
labels: ["specification", "RAG", "chatbot", "docusaurus", "ai"]
links:
  spec: ../specs/002-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/002-rag-chatbot/spec.md
 - specs/002-rag-chatbot/checklists/requirements.md
tests:
 - none
---

## Prompt

/sp.specify Build a fully-integrated RAG Chatbot for my Docusaurus Book

Target audience:
- Developers and AI engineers reading an AI/Robotics textbook on a Docusaurus site
- Students who want inline Q&A on book chapters
- Hackathon evaluators reviewing technical competence and scalability

Project Goal:
Create a production-ready Retrieval-Augmented Generation (RAG) chatbot embedded inside a Docusaurus book website. It must support:
1) Normal Q&A about the book
2) "Ask about selected text" mode — answers restricted ONLY to user-selected text
3) Streaming responses
4) Full backend using FastAPI + Qdrant + Cohere embeddings
5) Gemini model via OpenAI Agents SDK
6) Optional Context7 MCP integration for pulling up-to-date API docs
7) Clean reusable code structure for future agent sub-skills

Success Criteria:
- A working chat widget permanently visible on all Docusaurus pages
- Users can select text → click "Ask about selection" → get an answer using solely that text
- RAG pipeline: Docusaurus markdown is chunked, embedded with Cohere, stored in Qdrant, and retrieved
- Backend exposes: /ingest, /search, /chat
- Chat endpoint:
  - handles selected-text mode
  - handles top-k retrieval mode
  - constructs a strict context-only prompt
  - sends request to Gemini using OpenAI Agents SDK Runner
- End-to-end working demo deployed to GitHub Pages + Cloud backend
- Code organized into uv project layout
- Logging + error handling + CORS configured
- Fully reproducible setup instructions generated

Constraints:
- Language: TypeScript for Docusaurus UI, Python for backend
- Embedding model: Cohere embed-v4.0 (or multilingual if needed)
- Vector DB: Qdrant Cloud Free Tier
- LLM: Gemini 2.0 Flash via OpenAI Agents SDK (Runner API)
- Backend: FastAPI with Uvicorn
- Must work offline locally with dotenv and online with environment variables
- Docusaurus must remain static; chatbot communicates via REST API only
- API must avoid hallucination by enforcing context-bound answers

Not building:
- A full-featured chat "app"
- Fine-tuning or training a custom model
- Browser extensions or server-side rendering version
- A multi-tenant SaaS platform
- Full security/authorization system (beyond CORS & rate limits)

Additional Requirements:
- Generate all necessary files & code:
  - Chat UI component
  - Backend FastAPI project
  - Qdrant schema creation
  - Cohere embedding ingestion script
  - RAG pipeline code
  - Gemini Agent integration code
  - Docs for deployment
- Use Context7 MCP to fetch up-to-date docs for Cohere, Qdrant, and Agents SDK during coding
- Deliver step-by-step build instructions
- Provide well-named directories and modular structure usable by future Claude Code Subagents

Timeline:
- First usable version within 3 days
- Full production version within 7 days

## Response snapshot

Created a complete specification for the RAG Chatbot feature, including user scenarios, functional requirements, and success criteria.

## Outcome

- ✅ Impact: Successfully created a comprehensive specification for the RAG Chatbot feature with all required sections
- 🧪 Tests: none
- 📁 Files: specs/002-rag-chatbot/spec.md, specs/002-rag-chatbot/checklists/requirements.md
- 🔁 Next prompts: /sp.plan to create the implementation plan
- 🧠 Reflection: The specification clearly defines the scope and requirements for the RAG chatbot, balancing functionality with implementation constraints.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
