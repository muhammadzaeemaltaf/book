# ADR-0003: AI Summary Generation and Caching Strategy

> **Scope**: Document decision clusters, not individual technology choices. Group related decisions that work together (e.g., "Frontend Stack" not separate ADRs for framework, styling, deployment).

- **Status:** Accepted
- **Date:** 2025-12-12
- **Feature:** 003-auth-personalization
- **Context:** Need to implement AI-powered summary generation for educational content that is accessible only to authenticated users. The system must provide fast response times, control API costs, and maintain quality while handling variable request loads.

<!-- Significance checklist (ALL must be true to justify this ADR)
     1) Impact: Long-term consequence for architecture/platform/security?
     2) Alternatives: Multiple viable options considered with tradeoffs?
     3) Scope: Cross-cutting concern (not an isolated detail)?
     If any are false, prefer capturing as a PHR note instead of an ADR. -->

## Decision

- **AI Model**: OpenAI Agent SDK with Gemini model for summary generation
- **Caching Strategy**: Database caching with Neon Postgres for generated summaries
- **Cache Policy**: TTL-based expiration with 24-hour default, regeneration on content changes
- **Access Control**: AI summaries only available to authenticated users
- **Performance Target**: <30s generation time for initial request, <2s for cached summaries
- **Quality Control**: Content safety checks and validation before serving summaries
- **Usage Tracking**: Access count and timestamp tracking for cache optimization

## Consequences

### Positive

- Significant cost reduction through summary caching instead of repeated API calls
- Fast response times for users accessing previously generated summaries
- Maintains quality of AI-generated content through Gemini model
- Proper access control ensures summaries are only available to authenticated users
- Usage tracking enables optimization of caching strategy over time
- Content safety checks prevent inappropriate summaries from being served

### Negative

- Complexity of cache invalidation when source content changes
- Storage costs for cached summaries in database
- Potential staleness of cached content if source material updates
- Dependency on external AI service availability and performance
- Additional database load from storing and retrieving cached summaries

## Alternatives Considered

- **No Caching**: Generate summaries on-demand every time - Rejected due to high API costs and poor performance
- **Redis Caching**: Use Redis for summary caching instead of database - Rejected due to additional infrastructure complexity when Neon Postgres already provides adequate caching capability
- **Client-Side Caching**: Browser-based caching - Rejected due to security concerns (summaries should only be accessible to authenticated users)
- **Different AI Models**: Alternative models like GPT-4 or Claude - Rejected as Gemini was specified in requirements
- **Pre-generation**: Generate all summaries ahead of time - Rejected due to storage waste for potentially unused summaries and inability to adapt to content changes

## References

- Feature Spec: specs/003-auth-personalization/spec.md
- Implementation Plan: specs/003-auth-personalization/plan.md
- Related ADRs: ADR-002 (Technology Stack Ubuntu ROS2), ADR-004 (RAG Chatbot Architecture)
- Evaluator Evidence: history/prompts/003-auth-personalization/1-auth-personalization-spec.spec.prompt.md, history/prompts/003-auth-personalization/2-auth-personalization-plan.plan.prompt.md
