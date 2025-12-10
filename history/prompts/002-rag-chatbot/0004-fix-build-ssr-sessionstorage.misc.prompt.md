---
id: 0004
title: fix-build-ssr-sessionstorage
stage: misc
date: 2025-12-10
surface: agent
model: claude-sonnet-4-5-20250929
feature: 002-rag-chatbot
branch: 002-rag-chatbot
user: muhammad-zaeem-altaf
command: /sp.analyze
labels: ["build", "ssr", "sessionstorage", "docusaurus", "fix"]
links:
  spec: ../specs/002-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: https://github.com/muhammadzaeemaltaf/book/pull/9
files:
  - src/components/ChatWidget.tsx
tests:
  - npm run build
---

## Prompt

run npm run build and fix build error

## Response snapshot

### Problem Identified
The build was failing with `ReferenceError: sessionStorage is not defined` for all 32 pages during static site generation (SSR). This occurred because `sessionStorage` is a browser-only API that's not available during Docusaurus's server-side rendering build process.

### Root Cause
In `ChatWidget.tsx`, `sessionStorage` was being accessed directly in multiple places:
1. In `useState` initializer for `isMinimized` state (line 34)
2. In `toggleChat` function (line 178)
3. In `toggleMinimize` function (line 199)
4. In `handleSelectText` function (lines 216, 223)

### Solution Applied
Wrapped all `sessionStorage` access with browser environment checks:
```typescript
if (typeof window !== 'undefined' && typeof sessionStorage !== 'undefined') {
  // sessionStorage operations
}
```

This ensures:
- SSR build completes without errors (returns default values during build)
- Browser runtime works normally (persists state in sessionStorage)
- Progressive enhancement (graceful degradation if sessionStorage unavailable)

### Changes Made
Updated `src/components/ChatWidget.tsx`:
1. Added browser check in `isMinimized` state initializer
2. Wrapped `sessionStorage.setItem()` calls in `toggleChat()`
3. Wrapped `sessionStorage.setItem()` calls in `toggleMinimize()`
4. Wrapped `sessionStorage.setItem()` calls in `handleSelectText()`

### Build Result
✅ Build succeeded with static files generated in `build/` directory
- 80 pages compiled successfully
- Client bundle: 13.51s
- Server bundle: 4.97s

Remaining warnings (non-blocking):
- Deprecated `onBrokenMarkdownLinks` config option
- Duplicate route at `/docs/intro`
- Broken relative links in appendices page

## Outcome

- ✅ Impact: Fixed critical SSR build failure affecting all 32 pages; build now succeeds and generates static site
- 🧪 Tests: Build test passed with `npm run build` completing successfully
- 📁 Files: Updated src/components/ChatWidget.tsx with browser environment checks
- 🔁 Next prompts: Optional - fix remaining warnings (deprecated config, duplicate routes, broken links)
- 🧠 Reflection: Common SSR issue when using browser APIs; always check for `typeof window !== 'undefined'` before accessing browser-only APIs in React components used with SSR frameworks like Docusaurus

## Evaluation notes (flywheel)

- Failure modes observed: Browser API access during SSR causes build failures
- Graders run and results (PASS/FAIL): npm run build - PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Consider abstracting browser check into a custom hook like `useSafeSessionStorage()` for reusability
