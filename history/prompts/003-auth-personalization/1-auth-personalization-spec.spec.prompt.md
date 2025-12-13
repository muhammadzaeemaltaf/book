---
id: 1
title: auth-personalization-spec
stage: spec
date: '2025-12-12'
surface: agent
model: claude-sonnet-4-5-20250929
feature: auth-personalization
branch: 003-auth-personalization
user: muhammad-zaeem-altaf
command: /sp.specify
labels:
  - authentication
  - personalization
  - ai-summaries
  - specification
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files: []
tests: []
---

# PHR: auth-personalization-spec

## Outcome
Created comprehensive feature specification for authentication, personalization, and AI summaries feature.

## Evaluation
Specification includes user scenarios, functional requirements, success criteria, and key entities for implementing BetterAuth authentication, user background collection, personalized content, and AI-generated summaries.

## Prompt Text
```
Authentication + Personalization + AI Summaries

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
- FastAPI is already working we need to just add this additional count
```

## Response Text
Created comprehensive feature specification document at specs/003-auth-personalization/spec.md with:
- User scenarios and testing (3 prioritized user stories)
- Functional requirements (14 specific requirements)
- Key entities definition
- Success criteria with measurable outcomes
- Edge cases consideration
- Quality checklist validation