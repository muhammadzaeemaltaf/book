import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import { getApiUrl } from '../config/api';
import { setCookie, getCookie, deleteCookie } from '../utils/cookies';

// Auth service for handling session persistence and API calls

class AuthService {
  constructor() {
    this.tokenKey = 'access_token';
    this.userKey = 'user';
  }

  // Store authentication data in both localStorage and cookies
  setAuthData(token, user) {
    if (!ExecutionEnvironment.canUseDOM) {
      return;
    }
    if (token) {
      localStorage.setItem(this.tokenKey, token);
      setCookie(this.tokenKey, token, 7); // Expires in 7 days
    }
    if (user) {
      const userJson = JSON.stringify(user);
      localStorage.setItem(this.userKey, userJson);
      setCookie(this.userKey, encodeURIComponent(userJson), 7); // Expires in 7 days
    }
  }

  // Get stored token from cookie or localStorage
  getToken() {
    if (!ExecutionEnvironment.canUseDOM) {
      return null;
    }
    return getCookie(this.tokenKey) || localStorage.getItem(this.tokenKey);
  }

  // Get stored user from cookie or localStorage
  getUser() {
    if (!ExecutionEnvironment.canUseDOM) {
      return null;
    }
    let userStr = getCookie(this.userKey) || localStorage.getItem(this.userKey);
    if (!userStr) return null;
    
    try {
      // Decode if it's from cookie
      if (userStr && userStr.includes('%')) {
        userStr = decodeURIComponent(userStr);
      }
      return JSON.parse(userStr);
    } catch (error) {
      console.error('Error parsing user data:', error);
      return null;
    }
  }

  // Check if user is authenticated
  isAuthenticated() {
    if (!ExecutionEnvironment.canUseDOM) {
      return false;
    }
    const token = this.getToken();
    return !!token;
  }

  // Clear authentication data from both localStorage and cookies
  clearAuthData() {
    if (!ExecutionEnvironment.canUseDOM) {
      return;
    }
    localStorage.removeItem(this.tokenKey);
    localStorage.removeItem(this.userKey);
    deleteCookie(this.tokenKey);
    deleteCookie(this.userKey);
  }

  // Check if token is expired (in a real implementation, you'd decode the JWT to check expiration)
  isTokenExpired() {
    // For now, we'll assume the token doesn't expire
    // In a real implementation, you would decode the JWT and check the exp claim
    return false;
  }

  // Refresh token if needed (in a real implementation)
  async refreshToken() {
    if (!ExecutionEnvironment.canUseDOM) {
      throw new Error('Cannot refresh token during SSR');
    }
    // In a real implementation, you would call an API endpoint to refresh the token
    // This is a placeholder that would make an actual API call
    try {
      const refreshToken = localStorage.getItem('refresh_token');
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }

      const response = await fetch(getApiUrl('/api/auth/refresh'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refresh_token: refreshToken })
      });

      if (!response.ok) {
        throw new Error('Failed to refresh token');
      }

      const data = await response.json();
      this.setAuthData(data.access_token, this.getUser());
      return data.access_token;
    } catch (error) {
      console.error('Error refreshing token:', error);
      this.clearAuthData();
      throw error;
    }
  }

  // Make authenticated API request with automatic token refresh if needed
  async makeAuthenticatedRequest(url, options = {}) {
    let token = this.getToken();

    // If token is expired, try to refresh it
    if (this.isTokenExpired() && token) {
      try {
        token = await this.refreshToken();
      } catch (error) {
        // If refresh fails, clear auth data and throw error
        this.clearAuthData();
        throw error;
      }
    }

    // Add authorization header
    const headers = {
      ...options.headers,
      'Authorization': `Bearer ${token}`
    };

    // Make the request
    const response = await fetch(getApiUrl(url.startsWith('/') ? url : `/${url}`), {
      ...options,
      headers
    });

    // If we get a 401, clear auth data and redirect to login
    if (response.status === 401) {
      this.clearAuthData();
      window.location.href = '/signin';
      throw new Error('Authentication required');
    }

    return response;
  }

  // Signup API call
  async signup(userData) {
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

      // Store the authentication data if returned
      if (data.access_token && data.user) {
        this.setAuthData(data.access_token, data.user);
      }

      return {
        success: true,
        ...data
      };
    } catch (error) {
      return {
        success: false,
        message: error.message
      };
    }
  }

  // Signin API call
  async signin(credentials) {
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

      // Store the authentication data
      if (data.access_token && data.user) {
        this.setAuthData(data.access_token, data.user);
      }

      return {
        success: true,
        ...data
      };
    } catch (error) {
      return {
        success: false,
        message: error.message
      };
    }
  }

  // Signout API call
  async signout() {
    try {
      const token = this.getToken();
      if (token) {
        // Call the API to sign out (this might invalidate the token on the server)
        await fetch(getApiUrl('/api/auth/signout'), {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
      }
    } catch (error) {
      console.error('Error during signout API call:', error);
      // Continue with clearing local data even if API call fails
    } finally {
      // Clear local storage regardless of API call result
      this.clearAuthData();
    }
  }

  // Get current user info
  async getCurrentUser() {
    try {
      const token = this.getToken();
      if (!token) {
        return null;
      }

      const response = await this.makeAuthenticatedRequest('/api/auth/me');

      if (!response.ok) {
        throw new Error('Failed to get user info');
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting current user:', error);
      return null;
    }
  }

  // Update user profile
  async updateProfile(profileData) {
    try {
      const response = await this.makeAuthenticatedRequest('/api/user/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(profileData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to update profile');
      }

      return await response.json();
    } catch (error) {
      return {
        success: false,
        message: error.message
      };
    }
  }
}

// Create and export a singleton instance
const authService = new AuthService();
export default authService;