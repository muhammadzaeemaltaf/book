---
title: Frontend Platform Architecture - Docusaurus with GitHub Pages
status: Accepted
date: 2025-12-06
deciders: [Muhammad Zaeem Altaf]
consulted: []
informed: []
---

# ADR-001: Frontend Platform Architecture - Docusaurus with GitHub Pages

## Context

The Physical AI & Humanoid Robotics textbook requires a web-based delivery platform that can handle educational content with rich interactivity, including code examples, diagrams, and an integrated RAG chatbot. The platform must be accessible, maintainable, and support the progressive learning approach outlined in the project requirements. The solution needs to be cost-effective and easily deployable while supporting static content with interactive components.

## Decision

We will use Docusaurus 3.x as the static site generator with GitHub Pages for deployment. This decision includes:

- **Framework**: Docusaurus 3.x with Node.js 18+
- **Content Format**: Markdown with MDX for interactive components
- **Deployment**: GitHub Pages via GitHub Actions
- **Search**: Built-in Docusaurus search functionality
- **Custom Components**: Support for React components for RAG chatbot integration

## Alternatives Considered

1. **GitBook** - More limited customization options, less flexible for interactive components
2. **Hugo** - Requires more complex templating for interactive features, less suitable for educational content
3. **Next.js** - More complex setup than needed for documentation site, higher maintenance overhead
4. **Custom React App** - Would require significant development time for features Docusaurus provides out-of-the-box

## Consequences

### Positive
- Excellent Markdown support with MDX for interactive components (RAG chatbot)
- Built-in search functionality reduces development time
- GitHub Pages deployment is cost-effective and reliable
- Strong plugin ecosystem for documentation sites
- Supports versioning if needed for future textbook updates
- Progressive web app capabilities for offline reading
- Responsive design for mobile access

### Negative
- Learning curve for Docusaurus-specific features
- Potential limitations with complex interactive components
- Dependency on Docusaurus ecosystem for future maintenance
- May require custom theming to achieve desired educational UX

## References

- plan.md: Technical Context section
- research.md: Technology Stack Decisions - Docusaurus as Static Site Generator
- data-model.md: Technical Constraints - Compatible with Docusaurus 3.x
- quickstart.md: GitHub Pages deployment configuration