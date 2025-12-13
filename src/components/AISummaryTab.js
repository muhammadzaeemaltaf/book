import React, { useState, useEffect, useContext } from 'react';
import { AuthContext } from '../contexts/AuthContext';
import { getApiUrl } from '../config/api';

const AISummaryTab = ({ chapterId, className = '' }) => {
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [lastGenerated, setLastGenerated] = useState(null);

  // Check if context is available, otherwise fallback to localStorage
  const authContext = useContext(AuthContext);
  const isAuthenticated = authContext ? authContext.isAuthenticated :
    !!localStorage.getItem('access_token');

  // Fetch existing summary when component mounts
  useEffect(() => {
    if (!isAuthenticated || !chapterId) return;

    const fetchSummary = async () => {
      setLoading(true);
      setError('');

      try {
        // Use helper to compute API URL for summary GET
        const apiUrl = getApiUrl(`/api/summary/${chapterId}`);

        const response = await fetch(apiUrl, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
          }
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.message || 'Failed to get summary');
        }

        const data = await response.json();
        setSummary(data.summary);
        setLastGenerated(data.metadata?.generation_timestamp || null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchSummary();
  }, [chapterId, isAuthenticated]);

  const generateSummary = async () => {
    if (!isAuthenticated) {
      setError('Please sign in to generate AI summaries');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Use helper to compute API URL for summary POST
      const apiUrl = getApiUrl(`/api/summary/${chapterId}`);

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({
          chapter_id: chapterId,
          force_new: true // Force regeneration
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || 'Failed to generate summary');
      }

      const data = await response.json();
      setSummary(data.summary);
      setLastGenerated(data.metadata?.generation_timestamp || null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className={`border border-gray-200 rounded-lg p-6 ${className}`}>
        <div className="text-center py-8">
          <h3 className="text-lg font-medium text-gray-900 mb-2">AI Summary</h3>
          <p className="text-gray-600 mb-4">
            Sign in to access AI-generated summaries for this chapter.
          </p>
          <button
            onClick={() => window.location.href = '/signin'}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Sign In
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`border border-gray-200 rounded-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-800">AI Summary</h3>
        <button
          onClick={generateSummary}
          disabled={loading}
          className="bg-blue-600 text-white px-3 py-1 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 text-sm disabled:opacity-50"
        >
          {loading ? 'Generating...' : 'Regenerate'}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-md text-sm">
          {error}
        </div>
      )}

      {loading && !summary && (
        <div className="flex items-center justify-center py-8">
          <svg className="animate-spin h-6 w-6 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span className="ml-2">Generating AI summary...</span>
        </div>
      )}

      {summary && !loading && (
        <div className="prose max-w-none">
          <div className="bg-blue-50 p-4 rounded-md mb-4">
            <p className="text-sm text-blue-800">
              {lastGenerated ? `Generated on ${new Date(lastGenerated).toLocaleString()}` : 'AI-generated summary'}
            </p>
          </div>
          <div className="text-gray-700">
            {summary}
          </div>
        </div>
      )}

      {!summary && !loading && !error && (
        <div className="text-center py-8">
          <p className="text-gray-600 mb-4">
            No AI summary available yet. Click "Regenerate" to create one.
          </p>
          <button
            onClick={generateSummary}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Generate Summary
          </button>
        </div>
      )}
    </div>
  );
};

export default AISummaryTab;