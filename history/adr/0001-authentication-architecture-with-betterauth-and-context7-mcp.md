# ADR-0001: Authentication Architecture with BetterAuth and Context7 MCP

> **Scope**: Document decision clusters, not individual technology choices. Group related decisions that work together (e.g., "Frontend Stack" not separate ADRs for framework, styling, deployment).

- **Status:** Accepted
- **Date:** 2025-12-12
- **Feature:** 003-auth-personalization
- **Context:** Need to implement secure, maintainable authentication for the Physical AI textbook platform while integrating with existing Context7 MCP infrastructure. The solution must support email/password authentication without OAuth providers, maintain SEO for public content, and integrate with Docusaurus frontend.

<!-- Significance checklist (ALL must be true to justify this ADR)
     1) Impact: Long-term consequence for architecture/platform/security?
     2) Alternatives: Multiple viable options considered with tradeoffs?
     3) Scope: Cross-cutting concern (not an isolated detail)?
     If any are false, prefer capturing as a PHR note instead of an ADR. -->

## Decision

- **Authentication Provider**: BetterAuth (via Context7 MCP) for secure email/password authentication
- **User Management**: BetterAuth manages core user table (id, email, emailVerified, image, name)
- **Profile Extension**: Custom UserProfile table for technical background data
- **Integration Pattern**: REST API endpoints from Docusaurus frontend to BetterAuth backend
- **Session Management**: JWT-based tokens with client-side state management
- **Database**: Neon Postgres for user profile and personalization data

## Consequences

### Positive

- Leverages existing Context7 MCP infrastructure for authentication
- Well-maintained, secure authentication solution with proper session handling
- Maintains SEO for public content while restricting personalized features to authenticated users
- Clean separation between core user data (managed by BetterAuth) and profile data (custom tables)
- Supports the required technical background collection during signup
- Integrates well with FastAPI backend and Docusaurus frontend

### Negative

- Adds dependency on BetterAuth ecosystem and Context7 MCP
- Requires additional complexity for profile data synchronization
- Learning curve for team unfamiliar with BetterAuth
- Potential vendor lock-in to BetterAuth patterns and APIs
- Additional network calls for authentication state verification

## Alternatives Considered

- **Custom Authentication**: Build authentication from scratch with custom JWT implementation - Rejected due to security complexity, maintenance burden, and reinventing proven solutions
- **NextAuth.js**: Standard authentication library - Rejected as it's not well-suited for Docusaurus integration and doesn't leverage Context7 MCP
- **Firebase Auth**: Cloud-based authentication service - Rejected due to unnecessary complexity for simple email/password auth and potential vendor lock-in to Google ecosystem
- **Auth0**: Enterprise authentication service - Rejected due to cost considerations and over-engineering for this use case

## References

- Feature Spec: specs/003-auth-personalization/spec.md
- Implementation Plan: specs/003-auth-personalization/plan.md
- Related ADRs: ADR-006 (Context7 MCP Integration)
- Evaluator Evidence: history/prompts/003-auth-personalization/1-auth-personalization-spec.spec.prompt.md, history/prompts/003-auth-personalization/2-auth-personalization-plan.plan.prompt.md
