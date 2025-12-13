import React, { useState, useCallback, useContext } from 'react';
import { AuthContext } from '../contexts/AuthContext';
import { getApiUrl } from '../config/api';

const PersonalizeButton = ({ chapterId, onPersonalize, className = '' }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Check if context is available, otherwise fallback to localStorage
  const authContext = useContext(AuthContext);
  const isAuthenticated = authContext ? authContext.isAuthenticated :
    !!localStorage.getItem('access_token');

  const handlePersonalize = useCallback(async () => {
    if (!isAuthenticated) {
      setError('Please sign in to personalize content');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Use helper to compute the API URL
      const apiUrl = getApiUrl(`/api/personalize/${chapterId}`);

      // In a real implementation, this would call the API to generate personalized content
      // For now, we'll simulate the API call
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}` // Assuming token is stored in localStorage
        },
        body: JSON.stringify({
          chapter_id: chapterId,
          content: '', // This would be the chapter content in a real implementation
          target_level: 'adaptive' // Default to adaptive personalization
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || 'Failed to personalize content');
      }

      const data = await response.json();

      if (onPersonalize) {
        onPersonalize(data);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [chapterId, isAuthenticated, onPersonalize]);

  if (!isAuthenticated) {
    return (
      <button
        onClick={() => window.location.href = '/signin'}
        className={`bg-gray-400 text-white px-4 py-2 rounded-md cursor-not-allowed ${className} btn-primary`}
        disabled
      >
        Sign in to Personalize
      </button>
    );
  }

  return (
    <div className="space-y-2">
      <button
        onClick={handlePersonalize}
        disabled={loading}
        className={`w-full bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 ${className} btn-primary`}
      >
        {loading ? (
          <span className="flex items-center justify-center">
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Personalizing...
          </span>
        ) : (
          'Personalize This Chapter'
        )}
      </button>

      {error && (
        <div className="text-red-600 text-sm">
          {error}
        </div>
      )}
    </div>
  );
};

export default PersonalizeButton;