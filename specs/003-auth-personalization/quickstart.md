# Quickstart Guide: Authentication + Personalization + AI Summaries

## Overview
This guide provides instructions for setting up and running the authentication, personalization, and AI summaries feature in the Physical AI textbook project.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docusaurus 3.x
- Access to Neon Postgres database
- Access to OpenAI API for Gemini model
- BetterAuth configured via Context7 MCP

## Environment Setup

### 1. Database Configuration
Set up your Neon Postgres database and update the environment variables:

```bash
# In your .env file
DATABASE_URL="postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require"
```

### 2. API Keys
Configure the necessary API keys:

```bash
# In your .env file
OPENAI_API_KEY="your_openai_api_key"
GEMINI_API_KEY="your_gemini_api_key"  # If using Gemini directly
QDRANT_URL="your_qdrant_cloud_url"
QDRANT_API_KEY="your_qdrant_api_key"
```

### 3. BetterAuth Configuration
Ensure BetterAuth is properly configured via Context7 MCP with email/password authentication enabled.

## Backend Setup

### 1. Install Backend Dependencies
```bash
cd backend
uv sync
```

### 2. Run Database Migrations
```bash
cd backend
python -m src.utils.database_setup  # Or your migration command
```

### 3. Start Backend Server
```bash
cd backend
uv run python -m src.api.main  # Or your server start command
```

The backend should now be running on the configured port (typically 8000).

## Frontend Setup

### 1. Install Frontend Dependencies
```bash
npm install
```

### 2. Run Development Server
```bash
npm run start
```

## Key Integration Points

### 1. Adding Auth to Pages
To add authentication to a Docusaurus page:

```jsx
// In your page component
import { useAuth } from '@site/src/contexts/AuthContext';

function MyPage() {
  const { user, isAuthenticated, signIn, signOut } = useAuth();

  if (!isAuthenticated) {
    return <div>Please <a href="/auth/signin">sign in</a> to access this content</div>;
  }

  // Your authenticated content here
  return <div>Welcome, {user.name}!</div>;
}
```

### 2. Personalization Button
To add a personalization button to a chapter:

```jsx
// In your chapter component
import { usePersonalization } from '@site/src/services/personalization';

function ChapterContent({ chapterId }) {
  const { personalizeContent, loading } = usePersonalization();

  const handlePersonalize = async () => {
    const personalized = await personalizeContent(chapterId);
    // Update the content with personalized version
  };

  return (
    <div>
      <button onClick={handlePersonalize} disabled={loading}>
        {loading ? 'Personalizing...' : 'Personalize This Chapter'}
      </button>
      {/* Chapter content */}
    </div>
  );
}
```

### 3. AI Summary Tab
To add an AI summary tab to a chapter:

```jsx
// In your chapter component
import { useAISummary } from '@site/src/services/ai-summary';

function ChapterWithSummary({ chapterId }) {
  const { getSummary, loading } = useAISummary();
  const [summary, setSummary] = useState(null);

  const fetchSummary = async () => {
    const result = await getSummary(chapterId);
    setSummary(result.summary);
  };

  return (
    <div>
      <div role="tablist">
        <button role="tab" aria-selected={!summary}>Content</button>
        <button
          role="tab"
          aria-selected={!!summary}
          onClick={fetchSummary}
          disabled={loading}
        >
          {loading ? 'Generating...' : 'AI Summary'}
        </button>
      </div>
      {summary && <div className="summary-content">{summary}</div>}
    </div>
  );
}
```

## Testing the Feature

### 1. Authentication Flow
1. Navigate to `/auth/signup`
2. Create a new account with background survey
3. Verify you can sign in and out
4. Check that `/api/auth/me` returns your user information

### 2. Personalization
1. Sign in with a complete profile
2. Navigate to any chapter
3. Click the "Personalize This Chapter" button
4. Verify content adapts to your technical background

### 3. AI Summaries
1. Sign in to the application
2. Navigate to any chapter
3. Access the AI Summary tab
4. Verify you get a relevant summary
5. Try as an anonymous user to confirm access is restricted

## Configuration Options

### Personalization Complexity Levels
The system adapts content based on user experience levels:
- **Beginner**: More explanations, basic examples
- **Intermediate**: Balanced approach with practical examples
- **Advanced**: Concise content with complex examples
- **Expert**: Advanced concepts with minimal explanation

### Caching Configuration
AI summaries are cached to improve performance and reduce API costs:
- Cache TTL: 24 hours (configurable)
- Cache invalidation: When source content changes
- Cache size limits: Prevent unlimited storage

## Troubleshooting

### Common Issues

1. **Auth state not persisting**
   - Verify BetterAuth session configuration
   - Check browser storage settings

2. **Personalization not working**
   - Ensure user profile is complete
   - Verify backend personalization service is running

3. **AI summaries timing out**
   - Check API key validity
   - Verify OpenAI/Gemini service availability
   - Review rate limiting settings

### Debugging API Calls
Enable debug logging in your environment:
```bash
DEBUG=true
LOG_LEVEL=verbose
```

## Next Steps

1. Customize the personalization algorithm for your specific content
2. Adjust the background survey fields to match your audience
3. Fine-tune AI summary generation parameters
4. Add analytics to track feature usage
5. Implement additional authentication providers if needed