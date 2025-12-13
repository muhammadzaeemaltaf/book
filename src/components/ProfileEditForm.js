import React, { useState } from 'react';
import BackgroundSurvey from './BackgroundSurvey';

const ProfileEditForm = ({ initialData, onSave, onCancel, loading = false }) => {
  const [profileData, setProfileData] = useState({
    name: initialData.name || '',
    email: initialData.email || '',
    python_experience: initialData.python_experience || 'none',
    cpp_experience: initialData.cpp_experience || 'none',
    js_ts_experience: initialData.js_ts_experience || 'none',
    ai_ml_familiarity: initialData.ai_ml_familiarity || 'none',
    ros2_experience: initialData.ros2_experience || 'none',
    gpu_details: initialData.gpu_details || 'none',
    ram_capacity: initialData.ram_capacity || '4GB',
    operating_system: initialData.operating_system || 'linux',
    jetson_ownership: initialData.jetson_ownership || false,
    realsense_lidar_availability: initialData.realsense_lidar_availability || false
  });
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleProfileUpdate = (updatedProfile) => {
    setProfileData(prev => ({
      ...prev,
      ...updatedProfile
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('');
    setError('');

    try {
      await onSave(profileData);
      setMessage('Profile updated successfully!');
    } catch (err) {
      setError(err.message || 'Failed to update profile');
    }
  };

  const handleCancel = () => {
    setMessage('');
    setError('');
    onCancel();
  };

  return (
    <form onSubmit={handleSubmit}>
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

      <div className="mb-4">
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

      <div className="mb-4">
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

      <div className="flex justify-end space-x-3">
        <button
          type="button"
          onClick={handleCancel}
          className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50"
        >
          {loading ? 'Saving...' : 'Save Changes'}
        </button>
      </div>
    </form>
  );
};

export default ProfileEditForm;