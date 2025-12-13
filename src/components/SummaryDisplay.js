import React from 'react';

const SummaryDisplay = ({ summary, metadata, loading, error, onRegenerate, className = '' }) => {
  if (error) {
    return (
      <div className={`border border-red-200 bg-red-50 rounded-lg p-4 ${className}`}>
        <div className="text-red-800">
          <p className="font-medium">Error loading summary:</p>
          <p>{error}</p>
        </div>
        {onRegenerate && (
          <button
            onClick={onRegenerate}
            className="mt-2 bg-red-600 text-white px-3 py-1 rounded-md hover:bg-red-700 text-sm"
          >
            Try Again
          </button>
        )}
      </div>
    );
  }

  if (loading) {
    return (
      <div className={`border border-gray-200 bg-gray-50 rounded-lg p-4 ${className}`}>
        <div className="flex items-center justify-center py-8">
          <svg className="animate-spin h-6 w-6 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span className="ml-2">Generating summary...</span>
        </div>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className={`border border-gray-200 rounded-lg p-4 ${className}`}>
        <p className="text-gray-600 italic">No summary available.</p>
      </div>
    );
  }

  return (
    <div className={`border border-gray-200 rounded-lg p-4 ${className}`}>
      {metadata && metadata.generation_timestamp && (
        <div className="mb-3">
          <span className="text-xs text-gray-500">
            {metadata.cached ? 'Cached summary' : 'Generated'}{' '}
            {new Date(metadata.generation_timestamp).toLocaleString()}
          </span>
        </div>
      )}

      <div className="prose max-w-none">
        <div className="text-gray-700 whitespace-pre-wrap">
          {summary}
        </div>
      </div>

      {onRegenerate && (
        <div className="mt-4">
          <button
            onClick={onRegenerate}
            className="text-sm bg-blue-600 text-white px-3 py-1 rounded-md hover:bg-blue-700"
          >
            Regenerate Summary
          </button>
        </div>
      )}
    </div>
  );
};

export default SummaryDisplay;