# Research Document: Authentication + Personalization + AI Summaries

## Overview
This document captures research findings for implementing authentication, personalization, and AI summaries for the Physical AI textbook project. It addresses technical decisions, best practices, and integration patterns.

## Decision: BetterAuth Integration via Context7 MCP
**Rationale**: BetterAuth provides a secure, well-maintained authentication solution that integrates with the existing Context7 MCP infrastructure. It supports email/password authentication without requiring OAuth providers (as specified in constraints).

**Alternatives considered**:
- Custom authentication: Would require more security considerations and maintenance
- NextAuth.js: Not suitable for Docusaurus integration
- Firebase Auth: Would introduce unnecessary complexity for simple email/password auth

## Decision: Database Schema Design
**Rationale**: Using Neon Postgres with dedicated tables for users, profiles, and personalization allows for proper data separation and scaling. BetterAuth can handle the users table while we manage profiles and personalization separately.

**Tables**:
- `users`: Managed by BetterAuth (id, email, emailVerified, image, name)
- `user_profiles`: Custom table with technical background fields (user_id, programming_experience, hardware_details, etc.)
- `personalization_records`: Links user profiles to chapters with personalized content
- `ai_summaries`: Cached summaries with access control

## Decision: Frontend Authentication State Management
**Rationale**: Using React Context API or Zustand for managing authentication state provides a clean, predictable way to handle auth state across the Docusaurus application while maintaining SEO for public content.

**Integration approach**: Create an AuthProvider component that wraps the application and provides auth state to components that need it.

## Decision: AI Summary Generation and Caching
**Rationale**: Using OpenAI Agent SDK with Gemini model for summary generation provides high-quality summaries. Caching prevents repeated API calls and reduces costs.

**Caching strategy**: Store generated summaries in the database with timestamps, regenerate only when source content changes or cache expires.

## Decision: Personalization Engine
**Rationale**: A backend service that takes user profile and chapter content to generate personalized content ensures consistent behavior and security.

**Approach**: Create a personalization service that analyzes user background and adapts content complexity, examples, and recommendations accordingly.

## Decision: API Contract Design
**Rationale**: RESTful API endpoints following standard patterns ensure consistency and maintainability.

**Endpoints to implement**:
- `POST /auth/signup` - User registration with background survey
- `POST /auth/signin` - User login
- `POST /auth/signout` - User logout
- `GET /auth/me` - Get current user info
- `PUT /user/profile` - Update user profile
- `POST /personalize/{chapter_id}` - Get personalized chapter content
- `GET /summary/{chapter_id}` - Get AI summary for chapter (auth required)

## Best Practices Researched

### Security Best Practices
- Implement proper input validation for background survey to prevent injection attacks
- Use HTTPS for all authentication endpoints
- Implement proper session management with secure cookies
- Validate JWT tokens on backend endpoints
- Sanitize user-generated content before display

### Performance Best Practices
- Implement proper caching for AI summaries to reduce API costs and response time
- Use database indexing on frequently queried fields
- Implement proper pagination for large datasets
- Optimize database queries to avoid N+1 problems

### Docusaurus Integration Best Practices
- Use MDX components for auth-aware content rendering
- Implement proper SEO handling to preserve public content visibility
- Use Docusaurus swizzling sparingly, prefer plugin approach where possible
- Maintain backward compatibility with existing content structure

## Integration Patterns

### BetterAuth + Docusaurus Integration
- Use BetterAuth's REST API endpoints from frontend
- Implement custom auth pages in Docusaurus (login, signup, profile)
- Use client-side session management with proper error handling

### FastAPI + Docusaurus Integration
- Deploy FastAPI backend separately with CORS configured for Docusaurus domain
- Use environment variables for API URL configuration
- Implement proper error handling and user feedback mechanisms

### Database Migration Strategy
- Use Alembic for database migrations
- Implement proper backup strategies before schema changes
- Plan for data migration if schema changes affect existing data