# ADR-0002: Backend Technology Stack for AI Personalization

> **Scope**: Document decision clusters, not individual technology choices. Group related decisions that work together (e.g., "Frontend Stack" not separate ADRs for framework, styling, deployment).

- **Status:** Accepted
- **Date:** 2025-12-12
- **Feature:** 003-auth-personalization
- **Context:** Need to implement backend services for user authentication, profile management, content personalization, and AI summary generation. The stack must integrate with existing Python/TypeScript ecosystem, support database operations with Neon Postgres, and provide reliable AI service integration.

<!-- Significance checklist (ALL must be true to justify this ADR)
     1) Impact: Long-term consequence for architecture/platform/security?
     2) Alternatives: Multiple viable options considered with tradeoffs?
     3) Scope: Cross-cutting concern (not an isolated detail)?
     If any are false, prefer capturing as a PHR note instead of an ADR. -->

## Decision

- **Web Framework**: FastAPI for backend API services with automatic OpenAPI documentation
- **Programming Language**: Python 3.10+ for backend services
- **Database**: Neon Postgres for user profiles, personalization records, and AI summaries
- **AI Integration**: OpenAI Agent SDK with Gemini model for AI summary generation
- **Caching**: Database-level caching for AI summaries to reduce API costs
- **Configuration**: Centralized configuration in utils/ directory with environment-based settings
- **Data Modeling**: SQLAlchemy/SQLModel for database models and migrations

## Consequences

### Positive

- FastAPI provides excellent developer experience with automatic API documentation
- Strong type hints and async support improve code quality and performance
- Neon Postgres offers managed PostgreSQL with good performance and scaling
- OpenAI Agent SDK with Gemini provides high-quality AI summary generation
- Python 3.10+ ecosystem has rich libraries for AI and data processing
- Integration with existing project technology stack (Python/TypeScript)

### Negative

- Learning curve for team members unfamiliar with FastAPI
- Potential vendor lock-in to OpenAI/Gemini APIs for AI functionality
- Additional complexity for database migration and management
- API costs for AI summary generation at scale
- Dependency on external AI service availability and rate limits

## Alternatives Considered

- **Django + PostgreSQL**: More full-featured but heavier than needed for this API-focused service - Rejected due to over-engineering concerns
- **Flask + PostgreSQL**: More familiar but lacks FastAPI's automatic documentation and type hinting features - Rejected due to less modern approach
- **Node.js/Express**: Would create technology fragmentation in backend ecosystem - Rejected to maintain consistency with existing Python backend
- **Different AI Services**: Alternatives like Cohere or Anthropic - Rejected as Gemini was specified in requirements
- **Different Database**: MongoDB or other NoSQL - Rejected due to need for relational data and existing Neon Postgres infrastructure

## References

- Feature Spec: specs/003-auth-personalization/spec.md
- Implementation Plan: specs/003-auth-personalization/plan.md
- Related ADRs: ADR-002 (Technology Stack Ubuntu ROS2), ADR-004 (RAG Chatbot Architecture)
- Evaluator Evidence: history/prompts/003-auth-personalization/1-auth-personalization-spec.spec.prompt.md, history/prompts/003-auth-personalization/2-auth-personalization-plan.plan.prompt.md
