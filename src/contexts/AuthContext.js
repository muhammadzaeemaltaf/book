import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { getApiUrl } from '../config/api';
import { setCookie, getCookie, deleteCookie } from '../utils/cookies';

// Auth context
const AuthContext = createContext();

// Auth reducer
const authReducer = (state, action) => {
  switch (action.type) {
    case 'SIGNUP_START':
      return { ...state, loading: true, error: null };
    case 'SIGNUP_SUCCESS':
      return {
        ...state,
        loading: false,
        isAuthenticated: true,
        user: action.payload.user,
        token: action.payload.token,
        error: null
      };
    case 'SIGNIN_START':
      return { ...state, loading: true, error: null };
    case 'SIGNIN_SUCCESS':
      return {
        ...state,
        loading: false,
        isAuthenticated: true,
        user: action.payload.user,
        token: action.payload.token,
        error: null
      };
    case 'SIGNOUT':
      return {
        ...state,
        isAuthenticated: false,
        user: null,
        token: null
      };
    case 'SET_USER':
      return {
        ...state,
        isAuthenticated: !!action.payload,
        user: action.payload
      };
    case 'SET_ERROR':
      return { ...state, loading: false, error: action.payload };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    default:
      return state;
  }
};

// Auth provider component
export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, {
    isAuthenticated: false,
    user: null,
    token: null,
    loading: true, // Start with loading true while we check for existing session
    error: null
  });

  // Check for existing session on app start
  useEffect(() => {
    const initAuth = async () => {
      // Try to get token from cookie first, then localStorage
      let token = getCookie('access_token') || localStorage.getItem('access_token');
      let user = getCookie('user') || localStorage.getItem('user');

      if (token && user) {
        try {
          // Decode user data if it's a string
          const parsedUser = typeof user === 'string' ? JSON.parse(decodeURIComponent(user)) : user;
          
          dispatch({
            type: 'SIGNIN_SUCCESS',
            payload: {
              user: parsedUser,
              token: token
            }
          });
        } catch (error) {
          console.error('Error parsing stored user data:', error);
          // Clear invalid stored data
          localStorage.removeItem('access_token');
          localStorage.removeItem('user');
          deleteCookie('access_token');
          deleteCookie('user');
        }
      }
      
      // Set loading to false after initialization
      dispatch({ type: 'SET_LOADING', payload: false });
    };

    initAuth();
  }, []);

  // Store user and token in both localStorage and cookies when they change
  useEffect(() => {
    if (state.token) {
      localStorage.setItem('access_token', state.token);
      setCookie('access_token', state.token, 7); // Expires in 7 days
    } else {
      localStorage.removeItem('access_token');
      deleteCookie('access_token');
    }

    if (state.user) {
      const userJson = JSON.stringify(state.user);
      localStorage.setItem('user', userJson);
      setCookie('user', encodeURIComponent(userJson), 7); // Expires in 7 days
    } else {
      localStorage.removeItem('user');
      deleteCookie('user');
    }
  }, [state.token, state.user]);

  // Signup function
  const signup = async (userData) => {
    dispatch({ type: 'SIGNUP_START' });

    try {
      const response = await fetch(getApiUrl('/api/auth/signup'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Signup failed');
      }

      // Use the actual API response data
      const result = {
        success: true,
        ...data  // This includes user, access_token, user_id, message, profile_created
      };

      dispatch({
        type: 'SIGNUP_SUCCESS',
        payload: {
          user: data.user,
          token: data.access_token
        }
      });

      return result;
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: error.message });
      return { success: false, message: error.message };
    }
  };

  // Signin function
  const signin = async (credentials) => {
    dispatch({ type: 'SIGNIN_START' });

    try {
      const response = await fetch(getApiUrl('/api/auth/signin'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Sign in failed');
      }

      // Use the actual API response data
      const result = {
        success: true,
        ...data  // This includes user, access_token, user_id, message
      };

      dispatch({
        type: 'SIGNIN_SUCCESS',
        payload: {
          user: data.user,
          token: data.access_token
        }
      });

      return result;
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: error.message });
      return { success: false, message: error.message };
    }
  };

  // Signout function
  const signout = async () => {
    try {
      // Call the API to sign out
      await fetch(getApiUrl('/api/auth/signout'), {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${state.token}`
        }
      });

      // Clear local storage, cookies, and state
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
      deleteCookie('access_token');
      deleteCookie('user');

      dispatch({ type: 'SIGNOUT' });
    } catch (error) {
      // Even if the API call fails, we should still clear local state
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
      deleteCookie('access_token');
      deleteCookie('user');
      dispatch({ type: 'SIGNOUT' });
    }
  };

  // Get current user info
  const getCurrentUser = async () => {
    if (!state.token) {
      return null;
    }

    try {
      const response = await fetch(getApiUrl('/api/auth/me'), {
        headers: {
          'Authorization': `Bearer ${state.token}`
        }
      });

      if (!response.ok) {
        // If the token is invalid, sign out the user
        if (response.status === 401) {
          signout();
        }
        return null;
      }

      const data = await response.json();

      // Return user data if authenticated, null otherwise
      if (data.authenticated) {
        return {
          id: data.user_id,
          email: data.email,
          name: data.name
        };
      } else {
        return null;
      }
    } catch (error) {
      console.error('Error fetching user info:', error);
      return null;
    }
  };

  const value = {
    ...state,
    signup,
    signin,
    signout,
    getCurrentUser
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    // Provide a fallback context with default values
    const getStoredToken = () => getCookie('access_token') || localStorage.getItem('access_token');
    const getStoredUser = () => {
      let userStr = getCookie('user') || localStorage.getItem('user');
      if (!userStr) return null;
      try {
        if (userStr.includes('%')) {
          userStr = decodeURIComponent(userStr);
        }
        return JSON.parse(userStr);
      } catch {
        return null;
      }
    };

    return {
      isAuthenticated: !!getStoredToken(),
      user: getStoredUser(),
      loading: false,
      error: null,
      signup: async (userData) => {
        try {
          const response = await fetch(getApiUrl('/api/auth/signup'), {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(userData)
          });

          const result = await response.json();

          if (!response.ok) {
            throw new Error(result.message || 'Signup failed');
          }

          // Store the authentication data if returned
          if (result.access_token && result.user) {
            localStorage.setItem('access_token', result.access_token);
            localStorage.setItem('user', JSON.stringify(result.user));
            setCookie('access_token', result.access_token, 7);
            setCookie('user', encodeURIComponent(JSON.stringify(result.user)), 7);
          }

          return { success: true, ...result };
        } catch (error) {
          return { success: false, message: error.message };
        }
      },
      signin: async (credentials) => {
        try {
          const response = await fetch(getApiUrl('/api/auth/signin'), {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(credentials)
          });

          const result = await response.json();

          if (!response.ok) {
            throw new Error(result.message || 'Sign in failed');
          }

          // Store the authentication data if returned
          if (result.access_token && result.user) {
            localStorage.setItem('access_token', result.access_token);
            localStorage.setItem('user', JSON.stringify(result.user));
            setCookie('access_token', result.access_token, 7);
            setCookie('user', encodeURIComponent(JSON.stringify(result.user)), 7);
          }

          return { success: true, ...result };
        } catch (error) {
          return { success: false, message: error.message };
        }
      },
      signout: async () => {
        try {
          // Call the API to sign out
          await fetch(getApiUrl('/api/auth/signout'), {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${getStoredToken()}`
            }
          });
        } catch (error) {
          // Continue with clearing local storage even if API call fails
        } finally {
          // Clear local storage and cookies
          localStorage.removeItem('access_token');
          localStorage.removeItem('user');
          deleteCookie('access_token');
          deleteCookie('user');
        }
      },
      getCurrentUser: async () => {
        const token = getStoredToken();
        if (!token) {
          return null;
        }

        try {
          const response = await fetch(getApiUrl('/api/auth/me'), {
            headers: {
              'Authorization': `Bearer ${token}`
            }
          });

          if (!response.ok) {
            if (response.status === 401) {
              // Clear invalid token
              localStorage.removeItem('access_token');
              localStorage.removeItem('user');
              deleteCookie('access_token');
              deleteCookie('user');
            }
            return null;
          }

          const data = await response.json();

          // Return user data if authenticated, null otherwise
          if (data.authenticated) {
            return {
              id: data.user_id,
              email: data.email,
              name: data.name
            };
          } else {
            return null;
          }
        } catch (error) {
          console.error('Error fetching user info:', error);
          return null;
        }
      }
    };
  }
  return context;
};

export { AuthContext };
export default AuthContext;