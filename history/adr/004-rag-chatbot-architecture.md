---
title: RAG Chatbot Architecture - OpenAI, Qdrant, and FastAPI
status: Accepted
date: 2025-12-06
deciders: [Muhammad Zaeem Altaf]
consulted: []
informed: []
---

# ADR-004: RAG Chatbot Architecture - OpenAI, Qdrant, and FastAPI

## Context

The Physical AI & Humanoid Robotics textbook requires an AI-assisted learning component that can answer questions about textbook content with relevant context. The RAG (Retrieval-Augmented Generation) system must be scalable, performant, and integrate well with the Docusaurus-based frontend. The solution needs to provide accurate responses while maintaining cost-effectiveness and supporting educational use cases.

## Decision

We will implement the RAG chatbot using OpenAI embeddings + Qdrant vector database + FastAPI backend. This decision includes:

- **Embeddings**: OpenAI text-embedding-3-small model for content vectorization
- **Vector Storage**: Qdrant Cloud for vector similarity search
- **Backend**: FastAPI for API endpoints and response generation
- **Integration**: Docusaurus chatbot component for textbook pages
- **Session Management**: Conversation history tracking with session IDs
- **API Contract**: Defined in textbook-rag-api.yaml with 5 core endpoints

## Alternatives Considered

1. **OpenAI + Pinecone** - More expensive, less control over vector operations
2. **Self-hosted embeddings (Sentence Transformers)** - Higher computational requirements, slower response times
3. **Different vector databases (Weaviate, Milvus)** - Less familiarity, different API patterns
4. **Different backend frameworks (Express, Flask)** - Less async support, fewer features
5. **Pre-built chatbot services** - Less customization for educational content

## Consequences

### Positive
- Enables AI-assisted learning experiences as required
- Scalable solution for content retrieval with high performance
- Integrates well with Docusaurus via custom components
- Supports Context7 MCP protocol for enhanced AI interactions
- Proven technology stack with good documentation and community
- Cost-effective for educational use with usage-based pricing

### Negative
- Dependency on external APIs (OpenAI, Qdrant) for core functionality
- Potential costs at scale for API usage
- Requires API key management and security considerations
- Vendor lock-in with specific embedding and vector database providers
- Additional complexity for offline deployment scenarios

## References

- plan.md: Technical Context section on RAG integration
- research.md: Technology Stack Decisions - RAG Chatbot Implementation
- contracts/textbook-rag-api.yaml: Complete API specification
- data-model.md: Technical constraints for RAG integration