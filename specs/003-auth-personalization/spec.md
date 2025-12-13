# Feature Specification: Authentication + Personalization + AI Summaries

**Feature Branch**: `003-auth-personalization`
**Created**: 2025-12-12
**Status**: Draft
**Input**: User description: "Authentication + Personalization + AI Summaries

## Goal
Implement authentication using **BetterAuth via Context7 MCP**, collect user hardware/software background, unlock personalized content, and provide an AI Summarization tab (available only to logged-in users). Summaries must be generated using the **OpenAI Agent SDK with the Gemini model**.

## Target Audience
Students, engineers, and developers using the Physical AI & Humanoid Robotics AI-Native book who benefit from personalized content and AI-generated summaries.

## Focus Areas

### 1. Authentication (BetterAuth + Context7 MCP)
- Use **BetterAuth** integrated through **Context7 MCP** (already configured in Claude CLI).
- Implement Signup, Signin, Signout.
- Auth state available in Docusaurus frontend.
- Backend session validation via FastAPI.
- Provide `/auth/me` to fetch active user.

### 2. User Background Survey at Signup
Collect:
- Python, C/C++, JS/TS experience
- AI/ML familiarity
- ROS2 / Robotics exposure
- GPU details (None / 1650 / 3050+ / 4070+ / Cloud GPU)
- RAM capacity
- OS (Linux / Windows / Mac)
- Jetson ownership
- RealSense / LiDAR availability

All stored in **Neon Postgres** under `user_profiles`.

### 3. Database (Neon Postgres)
Tables required:
- `users`
- `user_profiles`
- `user_personalization`
- `sessions` (only if BetterAuth requires persistence)

### 4. Personalized Chapter Content
- Display a **“Personalize This Chapter”** button for logged-in users.
- Button sends user profile + chapter context → backend personalization agent.
- Backend adapts chapter complexity, examples, and recommendations.

### 5. AI Summarization Tab (Locked for Guests)
- Each chapter includes an **AI Summary** tab.
- Locked unless user is logged in.
- Summaries generated via **OpenAI Agent SDK + Gemini model**.
- Summaries cached for performance.

## Success Criteria
- Fully working BetterAuth (Context7 MCP) authentication.
- Signup collects background survey.
- Background stored in Neon Postgres.
- Auth-aware UI in Docusaurus.
- Personalized content and AI Summaries work correctly.
- AI Summaries only available to logged-in users.

## Constraints
- Must use **BetterAuth via Context7 MCP**.
- Must use FastAPI backend + Neon Postgres.
- Must preserve SEO.
- Deliverables must be Markdown + working code.

## Not Building
- No OAuth providers.
- No admin panel.
- No password reset.
- No authoring tools.

## Dependencies
- BetterAuth + Context7 MCP
- Docusaurus
- FastAPI
- Neon Postgres
- OpenAI Agent SDK (Gemini)
- Qdrant
- Zustand/Context API

## Deliverables
- BetterAuth integration
- Signup/Signin/Logout UI
- Background survey modal/page
- FastAPI endpoints
- DB schema
- Personalized Chapter Button
- AI Summary tab (locked for guests)
- Deployment-ready implementation

## Caution
- This new feature does not ruin working backend functionality but start working with it.
- FastAPI is already working we need to just add this additional count"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Anonymous User Signs Up and Provides Background Information (Priority: P1)

A new visitor wants to create an account and provide their technical background to receive personalized content. They navigate to the signup page, create credentials, and complete a background survey about their programming experience, hardware setup, and robotics knowledge.

**Why this priority**: This is the foundational user journey that enables all other personalized features. Without this, users cannot access the core value proposition of personalized content.

**Independent Test**: A new user can complete the signup process with background survey and successfully log in, demonstrating that authentication and user profiling work as expected.

**Acceptance Scenarios**:

1. **Given** an anonymous user visits the site, **When** they click the signup button and complete the registration process with background survey, **Then** they are authenticated and their profile information is stored.

2. **Given** a user is filling out the background survey during signup, **When** they submit incomplete information, **Then** they receive appropriate validation feedback.

---
### User Story 2 - Authenticated User Accesses Personalized Chapter Content (Priority: P1)

An authenticated user visits a chapter page and clicks the "Personalize This Chapter" button to receive content adapted to their technical background and experience level. The system analyzes their profile and generates personalized examples and explanations.

**Why this priority**: This delivers the core value proposition of personalized learning content based on user's technical background.

**Independent Test**: An authenticated user can click the personalize button and see content that reflects their technical background, demonstrating the personalization engine works.

**Acceptance Scenarios**:

1. **Given** an authenticated user with a complete profile visits a chapter page, **When** they click the "Personalize This Chapter" button, **Then** the content adapts based on their technical background.

2. **Given** an authenticated user without a complete profile, **When** they attempt to personalize content, **Then** they are prompted to complete their profile first.

---
### User Story 3 - Authenticated User Accesses AI-Generated Summaries (Priority: P1)

An authenticated user navigates to a chapter and accesses the AI Summary tab to get a condensed version of the content. The summary is generated using AI based on the chapter content and potentially the user's background.

**Why this priority**: This provides additional value to authenticated users by offering AI-generated summaries, which is a key differentiator.

**Independent Test**: An authenticated user can access the AI Summary tab and receive a relevant summary, demonstrating the AI summarization feature works.

**Acceptance Scenarios**:

1. **Given** an authenticated user visits a chapter page, **When** they click on the AI Summary tab, **Then** they see a relevant summary of the chapter content.

2. **Given** an anonymous user visits a chapter page, **When** they attempt to access the AI Summary tab, **Then** they are prompted to authenticate first.

---
### User Story 4 - User Manages Authentication State (Priority: P2)

An authenticated user can sign out of their account and return to guest status, or remain logged in across browser sessions. They can also view their profile information.

**Why this priority**: Essential authentication functionality that ensures users can securely manage their session state.

**Independent Test**: A user can sign in, verify they're authenticated, sign out, and verify they're returned to guest status.

**Acceptance Scenarios**:

1. **Given** an authenticated user, **When** they click the signout button, **Then** they are logged out and their authentication state is cleared.

2. **Given** an anonymous user, **When** they enter valid credentials, **Then** they are authenticated and gain access to protected features.

---
### User Story 5 - User Updates Their Background Profile (Priority: P3)

An authenticated user can update their technical background information after initial signup to reflect changes in their skills or hardware setup, affecting future personalization.

**Why this priority**: Allows users to refine their profile over time, improving the personalization experience.

**Independent Test**: An authenticated user can access their profile and update background information, with changes reflected in subsequent personalization.

**Acceptance Scenarios**:

1. **Given** an authenticated user with an existing profile, **When** they update their background information, **Then** the changes are saved and affect future personalization.

---
### Edge Cases

- What happens when an AI summary generation request times out?
- How does the system handle users with minimal background information when personalizing content?
- What occurs when the AI summarization service is temporarily unavailable?
- How does the system handle invalid or malicious input during the background survey?
- What happens when database connection fails during authentication?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST implement user authentication using BetterAuth via Context7 MCP
- **FR-002**: System MUST provide signup, signin, and signout functionality for users
- **FR-003**: System MUST collect user technical background during signup including programming experience, hardware details, and robotics knowledge
- **FR-004**: System MUST store user profiles in Neon Postgres database under `user_profiles` table
- **FR-005**: System MUST provide a "Personalize This Chapter" button for authenticated users
- **FR-006**: System MUST adapt chapter content based on user's technical background and experience level
- **FR-007**: System MUST provide an AI Summary tab that is locked for anonymous users
- **FR-008**: System MUST generate AI summaries using OpenAI Agent SDK with Gemini model
- **FR-009**: System MUST cache AI summaries to improve performance and reduce API costs
- **FR-010**: System MUST expose an `/auth/me` endpoint to fetch active user information
- **FR-011**: System MUST validate user sessions on backend via FastAPI
- **FR-012**: System MUST preserve SEO for public content while restricting personalized features to authenticated users
- **FR-013**: System MUST provide appropriate UI indicators showing authentication state
- **FR-014**: System MUST validate all user input during background survey to prevent injection attacks

### Key Entities *(include if feature involves data)*

- **User**: Represents an authenticated user with credentials managed by BetterAuth, including basic account information
- **UserProfile**: Contains technical background information including programming experience (Python, C/C++, JS/TS), AI/ML familiarity, ROS2/Robotics exposure, hardware specifications (GPU, RAM, OS), and equipment ownership (Jetson, RealSense, LiDAR)
- **PersonalizationRecord**: Links user profiles to specific chapters with personalized content adaptations
- **AISummary**: Cached AI-generated summaries for chapters, linked to user access permissions

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can complete the signup process including background survey in under 3 minutes
- **SC-002**: 95% of authenticated users can successfully access personalized chapter content within 10 seconds of clicking the personalize button
- **SC-003**: 90% of users can generate AI summaries within 30 seconds of requesting them
- **SC-004**: AI Summary tab is accessible only to authenticated users (0% success rate for anonymous users attempting access)
- **SC-005**: System maintains SEO for public content while successfully restricting personalized features to authenticated users
- **SC-006**: Authentication state persists across browser sessions for 30 days unless explicitly logged out
- **SC-007**: Background survey collects at least 80% of requested fields for new user registrations