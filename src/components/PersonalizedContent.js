import React, { useState, useEffect, useContext } from 'react';
import { AuthContext } from '../contexts/AuthContext';
import { getApiUrl } from '../config/api';

const PersonalizedContent = ({ chapterId, originalContent, className = '' }) => {
  const [personalizedContent, setPersonalizedContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showPersonalized, setShowPersonalized] = useState(false);

  // Check if context is available, otherwise fallback to localStorage
  const authContext = useContext(AuthContext);
  const isAuthenticated = authContext ? authContext.isAuthenticated :
    !!localStorage.getItem('access_token');

  // Fetch personalized content if user is authenticated and wants to see personalized version
  useEffect(() => {
    const fetchPersonalizedContent = async () => {
      if (!isAuthenticated || !showPersonalized || personalizedContent) {
        return;
      }

      setLoading(true);
      setError('');

      try {
        // Use getApiUrl helper to compute endpoint
        const apiUrl = getApiUrl('/api/personalize/get');

        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
          },
          body: JSON.stringify({
            chapter_id: chapterId
          })
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.message || 'Failed to get personalized content');
        }

        const data = await response.json();
        setPersonalizedContent(data.personalized_content);
      } catch (err) {
        setError(err.message);
        setShowPersonalized(false); // Fall back to original content
      } finally {
        setLoading(false);
      }
    };

    fetchPersonalizedContent();
  }, [chapterId, showPersonalized, personalizedContent, isAuthenticated]);

  const handleTogglePersonalization = () => {
    if (!isAuthenticated) {
      setError('Please sign in to access personalized content');
      return;
    }
    setShowPersonalized(!showPersonalized);
  };

  if (!isAuthenticated) {
    return (
      <div className={`border border-gray-200 rounded-lg p-4 ${className}`}>
        <div className="prose max-w-none">
          {originalContent}
        </div>
        <div className="mt-4 text-center">
          <button
            onClick={() => window.location.href = '/signin'}
            className="text-blue-600 hover:text-blue-800 font-medium"
          >
            Sign in to see personalized content
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`border border-gray-200 rounded-lg p-4 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-800">
          {showPersonalized ? 'Personalized Content' : 'Original Content'}
        </h3>
        <button
          onClick={handleTogglePersonalization}
          className="text-sm bg-blue-600 text-white px-3 py-1 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          {showPersonalized ? 'Show Original' : 'Show Personalized'}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-md text-sm">
          {error}
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-8">
          <svg className="animate-spin h-6 w-6 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span className="ml-2">Generating personalized content...</span>
        </div>
      )}

      {!loading && (
        <div className="prose max-w-none">
          {showPersonalized && personalizedContent ? (
            <div dangerouslySetInnerHTML={{ __html: personalizedContent }} />
          ) : (
            <div dangerouslySetInnerHTML={{ __html: originalContent }} />
          )}
        </div>
      )}

      {!showPersonalized && !personalizedContent && !loading && (
        <div className="mt-4 text-sm text-gray-600 italic">
          Personalized content is available when you toggle the "Show Personalized" button above.
          The content will be adapted based on your technical background profile.
        </div>
      )}
    </div>
  );
};

export default PersonalizedContent;