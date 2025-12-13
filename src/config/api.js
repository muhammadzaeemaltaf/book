// Helper to compute backend base URL safely in browser or server environments
export const getBackendBaseUrl = (fallbackPort = '8000') => {
  // Prefer explicit environment variables (support both REACT_APP_ and NEXT_PUBLIC_ variants)
  const envUrl =
    typeof process !== 'undefined' && process.env
      ? process.env.REACT_APP_BACKEND_URL ||
        process.env.NEXT_PUBLIC_BACKEND_URL ||
        process.env.REACT_APP_API_BASE_URL ||
        process.env.NEXT_PUBLIC_API_BASE_URL ||
        null
      : null;

  if (envUrl) {
    return envUrl;
  }

  // If running in a browser, construct a backend URL from the current origin and a backend port
  if (typeof window !== 'undefined' && window.location) {
    try {
      const url = new URL(window.location.href);
      // Keep hostname, set port to fallbackPort
      url.port = fallbackPort;
      return url.origin; // includes port if set
    } catch (err) {
      // If URL parsing fails, fall back to localhost with fallbackPort
      return `http://localhost:${fallbackPort}`;
    }
  }

  // Final fallback
  return `http://localhost:${fallbackPort}`;
};

// API configuration for the frontend
const API_CONFIG = {
  // Base URL for the backend API
  // In development, this might be different from production
  BASE_URL:
    (typeof process !== 'undefined' && process.env && (process.env.REACT_APP_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL)) ||
    getBackendBaseUrl(),

  // Timeout for API requests (in milliseconds)
  TIMEOUT: 30000,

  // Endpoints
  ENDPOINTS: {
    AUTH: {
      SIGNUP: '/api/auth/signup',
      SIGNIN: '/api/auth/signin',
      SIGNOUT: '/api/auth/signout',
      ME: '/api/auth/me'
    },
    PERSONALIZATION: {
      GENERATE: '/api/personalize/',
      GET: '/api/personalize/get'
    },
    SUMMARY: {
      GET: '/api/summary/',
      METADATA: '/api/summary/:id/metadata'
    },
    USER: {
      PROFILE: '/api/user/profile',
      COMPLETION: '/api/user/profile/completion'
    }
  }
};

// Helper function to construct API URLs
export const getApiUrl = (endpoint) => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};

export default API_CONFIG;