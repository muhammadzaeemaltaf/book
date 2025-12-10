---
id: 0002
title: Plan RAG Chatbot System
stage: plan
date: 2025-12-09
surface: agent
model: Claude Sonnet 4.5
feature: 002-rag-chatbot
branch: 002-rag-chatbot
user: Muhammad Zaeem Altaf
command: /sp.plan
labels: ["plan", "RAG", "chatbot", "architecture", "subagent", "docusaurus"]
links:
  spec: ../specs/002-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/002-rag-chatbot/plan.md
 - specs/002-rag-chatbot/research.md
 - specs/002-rag-chatbot/data-model.md
 - specs/002-rag-chatbot/quickstart.md
 - specs/002-rag-chatbot/contracts/api-contract-v1.yaml
tests:
 - none
---

## Prompt

Create a complete implementation plan for building a full RAG Chatbot system for my Docusaurus textbook, using the RAGRetrievalAgent subagent as a core component of the architecture.

Plan Requirements:

1. Architecture Sketch
   Include a full architecture overview covering:
   - Docusaurus frontend with persistent chat widget
   - Selection-aware mode ("Ask about selected text")
   - FastAPI backend with endpoints:
       • /ingest
       • /search
       • /chat
   - Qdrant Cloud vector DB (schema, collections, metadata)
   - Cohere embeddings (embed-v4.0 or multilingual)
   - Gemini 2.0 Flash via OpenAI Agents SDK Runner
   - Context7 MCP integration for live API docs retrieval
   - **Explicit integration of the RAGRetrievalAgent subagent**:
       • Used for query normalization
       • Used to choose retrieval mode (selected_only vs vector)
       • Used to perform ranking & filtering
       • Used to generate structured retrieval plans
   - Data flow diagram showing:
       Docusaurus → Backend → RAGRetrievalAgent → Qdrant/Cohere → PromptBuilder → Gemini → Frontend

2. Section Structure
   Organize the entire plan into phases:
   - Phase 1: Research & Requirements Verification
   - Phase 2: Core Architecture & Backend Foundation
   - Phase 3: Embedding & Ingestion Pipeline (Cohere → Qdrant)
   - Phase 4: Chat UI & Docusaurus Integration
   - Phase 5: RAG Pipeline Design
       • Describe how the RAGRetrievalAgent takes over retrieval duties
       • How the main agent delegates retrieval to the subagent
       • How selected-text mode bypasses vector DB
   - Phase 6: Context7 MCP Integration
   - Phase 7: Deployment Strategy (FastAPI → Cloud, GH Pages)
   - Phase 8: QA, Validation & Acceptance Review

3. Research Approach
   - Use research-concurrent methodology (research while planning)
   - For API details (Cohere, Qdrant, Agents SDK), query Context7 MCP
   - Document any API behaviors that affect implementation:
       • Cohere embedding format
       • Qdrant vector parameters
       • Agents Runner configuration
   - Identify risks:
       • Embedding dimension mismatch
       • Incorrect chunking size
       • Inconsistent prompt context
       • Latency issues in free-tier Qdrant
   - Provide mitigation strategies

4. Quality Validation
   Define acceptance criteria:
   - Chatbot answers ONLY from allowed context
   - Selected-text mode answers exclusively from the highlighted text
   - Retrieval relevance is high (top-k meaningfully ranked)
   - Latency < 1.5s for Qdrant search
   - Ingestion pipeline is repeatable and stable
   - UI is stable inside GH Pages static environment
   - No hallucinations when context missing
   - RAGRetrievalAgent returns structured retrieval plans correctly

   Include validation steps for:
   - Embedding correctness
   - Retrieval accuracy
   - Subagent decision-making (auto vs vector vs selected_only)
   - Full E2E test cases

5. Decisions Needing Documentation
   List all design decisions that must be explicitly documented.
   These include:
   - Why Cohere embed-v4.0 vs multilingual
   - Qdrant vector size and cosine distance choice
   - Chunk size (tokens, overlap)
   - How the RAGRetrievalAgent decides retrieval mode
   - How prompt construction integrates retrieval outputs
   - Runner.sync vs Runner.async for Gemini
   - Placement of hallucination guardrails
   - Whether to store raw text or cleaned text in Qdrant

   For each decision:
   - present options
   - tradeoffs
   - recommended choice

6. Testing Strategy
   Build a detailed test plan including:
   - Unit tests:
       • embedding generator
       • ingestion script
       • RAGRetrievalAgent behavior
       • Qdrant search wrapper
   - Integration tests:
       • chat endpoint end-to-end
       • selected-text mode
       • chunking consistency
   - Stress tests:
       • search latency under load
       • multiple chapter ingestion
   - Regression tests:
       • when book content updates
       • when embedding model updates
   - Final acceptance review based on hackathon scoring:
       • RAG Core (100 points)
       • Subagent & reusable intelligence (50 points)
       • Personalization & Urdu translation (50 points)

7. Technical Requirements
   - Use the RAGRetrievalAgent as the **official retrieval layer**.
     All retrieval passes MUST go through this subagent.
   - Follow APA citation style from Spec-Kit Plus Constitution.
   - Organize plan into phases:
       Research → Foundation → Analysis → Synthesis
   - Deliver actionable, detailed, engineering-focused content.

Purpose:
Produce a complete, structured, actionable execution plan for the RAG Chatbot project, explicitly integrating the RAGRetrievalAgent subagent as a core intelligence component.

## Response snapshot

Created a comprehensive implementation plan for the RAG Chatbot system, including architecture, technical context, project structure, research findings, data models, and API contracts with explicit integration of the RAGRetrievalAgent subagent.

## Outcome

- ✅ Impact: Successfully created a complete implementation plan with all required components and RAGRetrievalAgent integration
- 🧪 Tests: none
- 📁 Files: specs/002-rag-chatbot/plan.md, research.md, data-model.md, quickstart.md, contracts/api-contract-v1.yaml
- 🔁 Next prompts: /sp.tasks to generate implementation tasks
- 🧠 Reflection: The plan provides a solid foundation for implementing the RAG chatbot with proper separation of concerns and subagent architecture.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
