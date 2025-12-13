import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import { withAuthGuard } from '../utils/authGuard';
import BackgroundSurvey from '../components/BackgroundSurvey';
import authService from '../services/authService';

const ProfileContent = () => {
  const [profileData, setProfileData] = useState({
    name: '',
    email: '',
    python_experience: 'none',
    cpp_experience: 'none',
    js_ts_experience: 'none',
    ai_ml_familiarity: 'none',
    ros2_experience: 'none',
    gpu_details: 'none',
    ram_capacity: '4GB',
    operating_system: 'linux',
    jetson_ownership: false,
    realsense_lidar_availability: false
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  // Load profile data when component mounts
  useEffect(() => {
    const loadProfile = async () => {
      try {
        setLoading(true);
        const userData = await authService.getCurrentUser();

        if (userData) {
          // In a real implementation, we would fetch the full profile
          // For now, we'll use mock data or the user data we have
          setProfileData(prev => ({
            ...prev,
            name: userData.user?.name || userData.name || '',
            email: userData.user?.email || userData.email || '',
            // Set other profile fields as needed
          }));
        }
      } catch (err) {
        setError('Failed to load profile data');
        console.error('Error loading profile:', err);
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, []);

  const handleProfileUpdate = async (updatedProfile) => {
    setProfileData(prev => ({
      ...prev,
      ...updatedProfile
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setMessage('');
    setError('');

    try {
      const result = await authService.updateProfile(profileData);

      if (result.success) {
        setMessage('Profile updated successfully!');
      } else {
        setError(result.message || 'Failed to update profile');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while updating profile');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-3xl">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">Your Profile</h1>

        {message && (
          <div className="mb-4 p-3 bg-green-50 text-green-700 rounded-md">
            {message}
          </div>
        )}

        {error && (
          <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-md">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="mb-6">
            <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
              Full Name
            </label>
            <input
              type="text"
              id="name"
              value={profileData.name}
              onChange={(e) => setProfileData({...profileData, name: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          <div className="mb-6">
            <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
              Email Address
            </label>
            <input
              type="email"
              id="email"
              value={profileData.email}
              onChange={(e) => setProfileData({...profileData, email: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
              disabled
            />
            <p className="mt-1 text-xs text-gray-500">Email cannot be changed</p>
          </div>

          <div className="mb-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Technical Background</h2>
            <BackgroundSurvey
              initialData={profileData}
              onDataChange={handleProfileUpdate}
            />
          </div>

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={saving}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50"
            >
              {saving ? 'Saving...' : 'Save Profile'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

const AuthenticatedProfileContent = withAuthGuard(ProfileContent);

const ProfilePage = () => {
  return (
    <Layout title="Profile" description="User Profile">
      <BrowserOnly fallback={<div className="container mx-auto px-4 py-8"><div className="flex justify-center items-center h-64"><div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div></div></div>}>
        {() => <AuthenticatedProfileContent />}
      </BrowserOnly>
    </Layout>
  );
};

export default ProfilePage;