import React, { useContext } from 'react';
import { AuthContext } from '../contexts/AuthContext';

const AuthStatus = ({ className = '' }) => {
  const authContext = useContext(AuthContext);

  // Handle case where component is used outside of AuthProvider
  if (!authContext) {
    // Fallback to localStorage check
    const token = localStorage.getItem('access_token');
    const userStr = localStorage.getItem('user');

    if (token && userStr) {
      try {
        const user = JSON.parse(userStr);

        // Generate a simple avatar based on the user's name or email
        const getInitials = (name, email) => {
          if (name) {
            return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
          }
          if (email) {
            return email.charAt(0).toUpperCase();
          }
          return '?';
        };

        const initials = getInitials(user.name, user.email);

        // Generate a background color based on the user's ID for consistency
        const getAvatarColor = (id) => {
          if (!id) return 'bg-gray-200';

          // Simple hash function to generate consistent colors
          let hash = 0;
          for (let i = 0; i < id.length; i++) {
            hash = id.charCodeAt(i) + ((hash << 5) - hash);
          }

          const colors = [
            'bg-blue-500', 'bg-green-500', 'bg-purple-500',
            'bg-yellow-500', 'bg-red-500', 'bg-indigo-500'
          ];

          return colors[Math.abs(hash) % colors.length];
        };

        const avatarColor = getAvatarColor(user.id);

        return (
          <div className={`flex items-center space-x-4 ${className}`}>
            <div className="flex items-center space-x-2">
              <div className={`${avatarColor} w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium`}>
                {initials}
              </div>
              <div className="text-sm">
                <span className="text-gray-700">Welcome, {user.name || user.email.split('@')[0]}</span>
              </div>
            </div>
            <button
              onClick={() => {
                localStorage.removeItem('access_token');
                localStorage.removeItem('user');
                window.location.href = '/';
              }}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              Sign Out
            </button>
          </div>
        );
      } catch (e) {
        // Invalid user data in localStorage
      }
    }

    // Not authenticated
    return (
      <div className={`flex items-center space-x-4 ${className}`}>
        <a href="/signin" className="text-sm text-blue-600 hover:text-blue-800 font-medium">
          Sign In
        </a>
        <span className="text-gray-300">|</span>
        <a href="/signup" className="text-sm text-blue-600 hover:text-blue-800 font-medium">
          Sign Up
        </a>
      </div>
    );
  }

  const { isAuthenticated, user, loading, signout } = authContext;

  if (loading) {
    return (
      <div className={`flex items-center ${className}`}>
        <span className="text-sm text-gray-600">Loading...</span>
      </div>
    );
  }

  if (isAuthenticated && user) {
    // Generate a simple avatar based on the user's name or email
    const getInitials = (name, email) => {
      if (name) {
        return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
      }
      if (email) {
        return email.charAt(0).toUpperCase();
      }
      return '?';
    };

    const initials = getInitials(user.name, user.email);

    // Generate a background color based on the user's ID for consistency
    const getAvatarColor = (id) => {
      if (!id) return 'bg-gray-200';

      // Simple hash function to generate consistent colors
      let hash = 0;
      for (let i = 0; i < id.length; i++) {
        hash = id.charCodeAt(i) + ((hash << 5) - hash);
      }

      const colors = [
        'bg-blue-500', 'bg-green-500', 'bg-purple-500',
        'bg-yellow-500', 'bg-red-500', 'bg-indigo-500'
      ];

      return colors[Math.abs(hash) % colors.length];
    };

    const avatarColor = getAvatarColor(user.id);

    return (
      <div className={`flex items-center space-x-4 ${className}`}>
        <div className="flex items-center space-x-2">
          <div className={`${avatarColor} w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium`}>
            {initials}
          </div>
          <div className="text-sm">
            <span className="text-gray-700">Welcome, {user.name || user.email.split('@')[0]}</span>
          </div>
        </div>
        <button
          onClick={signout}
          className="text-sm text-blue-600 hover:text-blue-800 font-medium"
        >
          Sign Out
        </button>
      </div>
    );
  }

  return (
    <div className={`flex items-center space-x-4 ${className}`}>
      <a href="/signin" className="text-sm text-blue-600 hover:text-blue-800 font-medium">
        Sign In
      </a>
      <span className="text-gray-300">|</span>
      <a href="/signup" className="text-sm text-blue-600 hover:text-blue-800 font-medium">
        Sign Up
      </a>
    </div>
  );
};

export default AuthStatus;