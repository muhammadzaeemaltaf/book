import { useState, useCallback, useContext } from 'react';
import { AuthContext } from '../contexts/AuthContext';
import { getApiUrl } from '../config/api';

const useSignupForm = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: ''
  });
  const [backgroundData, setBackgroundData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [formStep, setFormStep] = useState(1); // 1 = basic info, 2 = background survey
  const authContext = useContext(AuthContext);
  const signup = authContext ? authContext.signup :
    async (userData) => {
      // Fallback implementation that calls the API directly
      try {
        // Use helper to compute API URL for signup
        const apiUrl = getApiUrl('/api/auth/signup');

        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(userData)
        });

        // Check if response is JSON before parsing
        const contentType = response.headers.get('content-type');
        let result;

        if (contentType && contentType.includes('application/json')) {
          result = await response.json();
        } else {
          // If not JSON, try to handle as text or return error
          const text = await response.text();
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${text}`);
          }
          result = { success: response.ok, message: text };
        }

        if (!response.ok) {
          throw new Error(result.message || `HTTP ${response.status}: Signup failed`);
        }

        // Store the authentication data if returned
        if (result.access_token && result.user) {
          localStorage.setItem('access_token', result.access_token);
          localStorage.setItem('user', JSON.stringify(result.user));
        }

        return { success: true, ...result };
      } catch (error) {
        // Check if this is the "Unexpected token '<'" error
        if (error.message.includes("Unexpected token '<'")) {
          return {
            success: false,
            message: 'API server is not responding correctly. Please check if the backend server is running.'
          };
        }
        return { success: false, message: error.message };
      }
    };

  const updateBasicInfo = useCallback((data) => {
    setFormData(prev => ({ ...prev, ...data }));
  }, []);

  const updateBackgroundInfo = useCallback((data) => {
    setBackgroundData(prev => ({ ...prev, ...data }));
  }, []);

  const goToNextStep = useCallback(() => {
    setFormStep(prev => prev + 1);
  }, []);

  const goToPreviousStep = useCallback(() => {
    setFormStep(prev => Math.max(1, prev - 1));
  }, []);

  const resetForm = useCallback(() => {
    setFormData({
      email: '',
      password: '',
      name: ''
    });
    setBackgroundData({});
    setFormStep(1);
    setError('');
    setSuccess(false);
  }, []);

  const submitSignup = useCallback(async () => {
    setLoading(true);
    setError('');

    try {
      const signupData = {
        ...formData,
        ...backgroundData
      };

      const result = await signup(signupData);
      if (result.success) {
        setSuccess(true);
        setTimeout(() => {
          window.location.href = '/'; // Redirect to home after successful signup
        }, 2000);
      } else {
        setError(result.message || 'Signup failed');
      }
    } catch (err) {
      setError(err.message || 'An error occurred during signup');
    } finally {
      setLoading(false);
    }
  }, [formData, backgroundData, signup]);

  return {
    formData,
    backgroundData,
    loading,
    error,
    success,
    formStep,
    updateBasicInfo,
    updateBackgroundInfo,
    goToNextStep,
    goToPreviousStep,
    resetForm,
    submitSignup
  };
};

export default useSignupForm;