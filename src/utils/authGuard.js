import React, { useEffect, useState } from 'react';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import authService from '../services/authService';

// Higher-order component for protecting routes that require authentication
export const withAuthGuard = (Component) => {
  return function AuthGuardComponent(props) {
    const [isChecking, setIsChecking] = useState(true);
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    useEffect(() => {
      if (ExecutionEnvironment.canUseDOM) {
        const authenticated = authService.isAuthenticated();
        setIsAuthenticated(authenticated);
        setIsChecking(false);

        if (!authenticated) {
          // Redirect to sign in page with return URL
          const returnUrl = window.location.pathname + window.location.search;
          window.location.href = `/signin?return=${encodeURIComponent(returnUrl)}`;
        }
      }
    }, []);

    if (isChecking || !ExecutionEnvironment.canUseDOM) {
      return null; // Return null during SSR or while checking
    }

    if (!isAuthenticated) {
      return null; // Return null while redirect happens
    }

    return <Component {...props} />;
  };
};

// Higher-order component for protecting routes that require unauthenticated access (e.g., sign in page)
export const withUnauthGuard = (Component) => {
  return function UnauthGuardComponent(props) {
    const [isChecking, setIsChecking] = useState(true);
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    useEffect(() => {
      if (ExecutionEnvironment.canUseDOM) {
        const authenticated = authService.isAuthenticated();
        setIsAuthenticated(authenticated);
        setIsChecking(false);

        if (authenticated) {
          // Redirect to dashboard or home page
          window.location.href = '/';
        }
      }
    }, []);

    if (isChecking || !ExecutionEnvironment.canUseDOM) {
      return null; // Return null during SSR or while checking
    }

    if (isAuthenticated) {
      return null; // Return null while redirect happens
    }

    return <Component {...props} />;
  };
};

// Hook for checking authentication status (can be used in functional components)
export const useAuthGuard = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    if (ExecutionEnvironment.canUseDOM) {
      setIsAuthenticated(authService.isAuthenticated());
    }
  }, []);

  return {
    isAuthenticated,
    checkAuth: () => ExecutionEnvironment.canUseDOM ? authService.isAuthenticated() : false,
    redirectIfUnauthenticated: (returnUrl) => {
      if (ExecutionEnvironment.canUseDOM && !isAuthenticated) {
        const url = returnUrl || window.location.pathname;
        window.location.href = `/signin?return=${encodeURIComponent(url)}`;
      }
    },
    redirectIfAuthenticated: (redirectTo = '/') => {
      if (ExecutionEnvironment.canUseDOM && isAuthenticated) {
        window.location.href = redirectTo;
      }
    }
  };
};

// Function to protect routes programmatically
export const protectRoute = (nextUrl) => {
  if (!ExecutionEnvironment.canUseDOM) {
    return false;
  }

  const isAuthenticated = authService.isAuthenticated();

  if (!isAuthenticated) {
    const returnUrl = nextUrl || window.location.pathname;
    window.location.href = `/signin?return=${encodeURIComponent(returnUrl)}`;
    return false;
  }

  return true;
};

// Function to check if user can access a specific feature
export const canAccessFeature = (featureName) => {
  const user = authService.getUser();

  // For now, we'll implement basic access control
  // In a real implementation, this would check user roles/permissions
  if (!user) {
    return false;
  }

  // All authenticated users can access all features for now
  // This could be extended to check specific permissions
  return true;
};

// Middleware function for protecting API routes on the frontend
export const authMiddleware = async (req, options = {}) => {
  const token = authService.getToken();

  if (!token) {
    throw new Error('Authentication required');
  }

  // Add authorization header
  const headers = {
    ...options.headers,
    'Authorization': `Bearer ${token}`
  };

  // Make the request with the token
  const response = await fetch(req, {
    ...options,
    headers
  });

  // If we get a 401, redirect to sign in
  if (response.status === 401) {
    authService.clearAuthData();
    window.location.href = '/signin';
    throw new Error('Authentication required');
  }

  return response;
};

export default {
  withAuthGuard,
  withUnauthGuard,
  useAuthGuard,
  protectRoute,
  canAccessFeature,
  authMiddleware
};