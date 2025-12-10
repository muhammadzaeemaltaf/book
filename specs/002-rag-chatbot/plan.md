# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

The RAG Chatbot implementation will create a production-ready retrieval-augmented generation system integrated into a Docusaurus textbook site. The system consists of a FastAPI backend handling document ingestion, embedding generation with Cohere, vector storage in Qdrant Cloud, and response generation via Gemini 2.0 Flash through the OpenAI Agents SDK. The frontend features a persistent chat widget with text selection capability for context-restricted Q&A. A dedicated RAGRetrievalAgent subagent will handle all retrieval operations including query normalization, mode selection (selected-only vs vector), ranking, filtering, and structured retrieval planning as specified in the requirements.

## Technical Context

**Language/Version**: Python 3.10+ (backend), TypeScript (Docusaurus frontend)
**Primary Dependencies**: FastAPI (backend), Docusaurus 3.x (frontend), Cohere embed-v4.0 (embeddings), Qdrant Cloud (vector DB), OpenAI Agents SDK (Gemini integration), uv (project management)
**Storage**: Qdrant Cloud vector database for embeddings, local storage for Docusaurus content
**Testing**: pytest for backend, Jest for frontend components
**Target Platform**: Web application (FastAPI backend + Docusaurus static frontend)
**Project Type**: Web (backend API + static frontend)
**Performance Goals**: <10 second response time for queries, <1.5s for Qdrant search latency, 95% accuracy in context restriction
**Constraints**: Must work with GitHub Pages (static frontend), Docusaurus remains static (no SSR), context-bound answers only (no hallucination), free-tier Qdrant limitations
**Scale/Scope**: Single textbook content, multiple concurrent users, streaming responses for improved UX

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Initial Compliance Verification

**Pedagogical Clarity**: ✅ The RAG chatbot will enhance learning by providing immediate, contextual answers to student questions, supporting progressive understanding of complex concepts.

**Hands-on Practicality**: ✅ The implementation includes practical components (embedding pipeline, API endpoints, UI integration) with actionable code examples.

**Technical Accuracy**: ✅ All technical components (FastAPI, Cohere, Qdrant, Gemini) will be implemented according to official documentation and verified through testing.

**Accessibility**: ✅ The chatbot interface will make complex textbook content more accessible by providing instant clarification on difficult concepts.

**Integration Focus**: ✅ The system integrates multiple technologies (frontend, backend, vector DB, LLM, embeddings) as a unified solution.

**Docusaurus Compatibility**: ✅ The persistent chat widget will be built specifically for Docusaurus compatibility with proper MDX support and static deployment.

**Quality Assurance**: ✅ Implementation will include testing strategies, error handling, and verification of all technical components.

### Post-Design Compliance Check

After Phase 1 design completion, all constitutional requirements continue to be satisfied. The architectural design with separate backend and frontend components maintains Docusaurus compatibility while providing the necessary functionality for the RAG system. The RAGRetrievalAgent subagent implementation enhances the integration focus by providing a dedicated retrieval layer that handles all context-aware operations as required.

## Project Structure

### Documentation (this feature)

```text
specs/002-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── chat.py              # Chat query/response models
│   │   ├── document.py          # Document chunk and embedding models
│   │   └── retrieval.py         # Retrieval models
│   ├── services/
│   │   ├── embedding_service.py # Cohere embedding operations
│   │   ├── qdrant_service.py    # Vector database operations
│   │   ├── ingestion_service.py # Document processing pipeline
│   │   ├── retrieval_service.py # RAGRetrievalAgent implementation
│   │   └── chat_service.py      # Chat and response generation
│   ├── api/
│   │   ├── main.py              # FastAPI app definition
│   │   ├── routes/
│   │   │   ├── ingest.py        # /ingest endpoint
│   │   │   ├── search.py        # /search endpoint
│   │   │   └── chat.py          # /chat endpoint
│   │   └── middleware/
│   │       └── cors.py          # CORS configuration
│   ├── agents/
│   │   └── rag_retrieval_agent.py # RAGRetrievalAgent subagent
│   └── utils/
│       ├── config.py            # Configuration management
│       ├── logging.py           # Logging utilities
│       └── validators.py        # Input validation utilities
├── tests/
│   ├── unit/
│   │   ├── test_embedding_service.py
│   │   ├── test_retrieval_service.py
│   │   └── test_agents/
│   │       └── test_rag_retrieval_agent.py
│   ├── integration/
│   │   ├── test_ingest_endpoint.py
│   │   ├── test_search_endpoint.py
│   │   └── test_chat_endpoint.py
│   └── contract/
│       └── test_api_contracts.py
├── requirements.txt
├── pyproject.toml
└── README.md

frontend/
├── src/
│   ├── components/
│   │   ├── ChatWidget.tsx       # Main chat interface component
│   │   ├── ChatMessage.tsx      # Individual message display
│   │   ├── ChatInput.tsx        # Input area with streaming support
│   │   └── SelectionHandler.tsx # Text selection and context mode
│   ├── services/
│   │   ├── apiClient.ts         # API communication layer
│   │   └── chatService.ts       # Chat business logic
│   ├── hooks/
│   │   └── useChat.ts           # Chat state management
│   └── styles/
│       └── chat.css             # Chat widget styling

.env.example                 # Environment variables template (root level)
.env                         # Actual environment file (root level, gitignored)
```

**Note:** The project uses a single `.env` file at the root level for both frontend and backend configuration. The backend's `config.py` automatically loads environment variables from the root `.env` file.
├── docusaurus.config.js         # Docusaurus configuration
├── sidebars.js                  # Navigation configuration
└── static/
    └── chat-icon.svg            # Chat widget icon

# Root level files
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore patterns
└── README.md                    # Project overview
```

**Structure Decision**: The architecture follows a web application pattern with separate backend (FastAPI) and frontend (Docusaurus) components to maintain separation of concerns. The backend handles all RAG operations, embeddings, and LLM integration, while the frontend provides the persistent chat widget integrated into the Docusaurus site. The RAGRetrievalAgent is implemented as a core service within the backend to handle all retrieval logic as specified in the requirements.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
