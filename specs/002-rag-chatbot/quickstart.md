# Quickstart Guide: RAG Chatbot for Docusaurus

## Prerequisites

- Python 3.10+ installed
- Node.js 18+ installed (for Docusaurus)
- uv package manager installed (`pip install uv`)
- Cohere API key
- Qdrant Cloud account and API key
- Google AI API key (for Gemini)

## Environment Setup

### 1. Clone and Initialize the Project

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Create environment file (single .env file in project root)
cp .env.example .env
```

### 2. Update Environment Variables

Edit the `.env` file in the project root with your API keys:

```env
# Backend Configuration
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_URL=your_qdrant_cluster_url

# LLM Configuration (Groq recommended)
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq

# Alternative: Gemini
GOOGLE_API_KEY=your_google_ai_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Alternative: OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Backend Settings
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
DEBUG=false

# CORS Settings
FRONTEND_URL=http://localhost:3000  # Adjust for your Docusaurus URL

# Frontend Configuration
REACT_APP_API_BASE_URL=http://localhost:8000
```

> **Note:** The project now uses a single `.env` file in the root directory for both frontend and backend configuration.

## Backend Setup

### 1. Navigate to Backend Directory

```bash
cd backend
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 3. Start the Backend Server

```bash
# Using uv
uv run python -m src.api.main

# Or using python directly
python -m src.api.main
```

The backend will start on `http://localhost:8000`

## Frontend (Docusaurus) Integration

### 1. Navigate to Frontend Directory

```bash
cd frontend
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Integrate Chat Widget

Add the chat widget to your Docusaurus site by modifying `docusaurus.config.js`:

```js
// In docusaurus.config.js
module.exports = {
  // ... existing configuration
  plugins: [
    // ... existing plugins
    // Add the chat widget plugin
    [
      '@docusaurus/plugin-content-blog',
      {
        id: 'chat-widget',
        path: 'src/components/ChatWidget',
        routeBasePath: 'chat',
      },
    ],
  ],
  themes: [
    // ... existing themes
  ],
};
```

### 4. Add Chat Widget to Layout

In your Docusaurus layout (usually `src/pages/index.js` or layout component):

```jsx
import ChatWidget from '../components/ChatWidget';

function Layout(props) {
  return (
    <>
      {/* Your existing layout */}
      <ChatWidget />
    </>
  );
}
```

## Initial Setup and Ingestion

### 1. Ingest Your Docusaurus Content

Once the backend is running, you can ingest your textbook content:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./docs",  // Path to your Docusaurus docs directory
    "chunk_size": 512,
    "overlap": 50
  }'
```

### 2. Verify Ingestion

Check the status of the ingestion:

```bash
curl http://localhost:8000/ingest/status
```

## Testing the Chat Functionality

### 1. Test the Chat Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the main concept of this textbook?",
    "mode": "normal_qa",
    "stream": true
  }'
```

### 2. Test Selected Text Mode

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain this concept in more detail",
    "mode": "selected_text",
    "selected_text": "This is the text the user selected on the page...",
    "stream": true
  }'
```

## Running the Full System

### 1. Start Backend Server

```bash
cd backend
uv run python -m src.api.main
```

### 2. Start Docusaurus Frontend

```bash
cd frontend
npm run start
```

The Docusaurus site will be available at `http://localhost:3000` with the chat widget integrated.

## API Endpoints

### Backend API

- `POST /ingest` - Ingest documents into vector database
- `GET /ingest/status` - Check ingestion status
- `POST /search` - Search documents (for debugging)
- `POST /chat` - Chat with the RAG system
- `GET /health` - Health check

### Frontend Integration

The chat widget communicates with the backend via the configured API endpoints.

## Troubleshooting

### Common Issues

1. **API Key Issues**: Verify all API keys in `.env` are correct
2. **CORS Errors**: Ensure `FRONTEND_URL` matches your Docusaurus URL
3. **Qdrant Connection**: Verify QDRANT_URL and API key are correct
4. **Embedding Issues**: Check Cohere API key and model availability

### Debugging Commands

Check backend logs:
```bash
uv run python -m src.api.main --debug
```

Test API endpoints directly:
```bash
curl http://localhost:8000/health
```

## Next Steps

1. Customize the chat widget UI to match your site's theme
2. Fine-tune chunk sizes and overlap for optimal retrieval
3. Add additional content sources to the ingestion pipeline
4. Implement analytics and usage tracking