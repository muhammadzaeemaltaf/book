import React, { useState, useEffect } from 'react';

const BackgroundSurvey = ({ initialData = {}, onDataChange }) => {
  const [formData, setFormData] = useState({
    python_experience: initialData.python_experience || 'none',
    cpp_experience: initialData.cpp_experience || 'none',
    js_ts_experience: initialData.js_ts_experience || 'none',
    ai_ml_familiarity: initialData.ai_ml_familiarity || 'none',
    ros2_experience: initialData.ros2_experience || 'none',
    gpu_details: initialData.gpu_details || 'none',
    ram_capacity: initialData.ram_capacity || '8GB',
    operating_system: initialData.operating_system || 'linux',
    jetson_ownership: initialData.jetson_ownership || false,
    realsense_lidar_availability: initialData.realsense_lidar_availability || false
  });

  const experienceLevels = [
    { value: 'none', label: 'None', emoji: '🆕' },
    { value: 'beginner', label: 'Beginner', emoji: '📚' },
    { value: 'intermediate', label: 'Intermediate', emoji: '💻' },
    { value: 'advanced', label: 'Advanced', emoji: '🚀' },
    { value: 'expert', label: 'Expert', emoji: '⭐' }
  ];

  const gpuOptions = [
    { value: 'none', label: 'None', desc: 'No GPU' },
    { value: '1650', label: 'Entry Level', desc: 'GTX 1650 or lower' },
    { value: '3050+', label: 'Mid Range', desc: 'RTX 3050 or higher' },
    { value: '4070+', label: 'High End', desc: 'RTX 4070 or higher' },
    { value: 'cloud_gpu', label: 'Cloud GPU', desc: 'AWS, GCP, etc.' }
  ];

  const ramOptions = [
    { value: '4GB', label: '4GB' },
    { value: '8GB', label: '8GB' },
    { value: '16GB', label: '16GB' },
    { value: '32GB', label: '32GB' },
    { value: '64GB+', label: '64GB+' }
  ];

  const osOptions = [
    { value: 'linux', label: 'Linux', emoji: '🐧' },
    { value: 'windows', label: 'Windows', emoji: '🪟' },
    { value: 'mac', label: 'macOS', emoji: '🍎' }
  ];

  useEffect(() => {
    onDataChange(formData);
  }, [formData]); // Remove onDataChange from dependencies to prevent infinite loop

  const handleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleCheckboxChange = (field, checked) => {
    setFormData(prev => ({
      ...prev,
      [field]: checked
    }));
  };

  const RadioGroup = ({ name, value, options, onChange, showEmoji = false }) => (
    <div className="survey-radio-group">
      {options.map((option) => {
        const isSelected = value === option.value;
        return (
          <div
            key={option.value}
            className={`survey-radio-option ${isSelected ? 'survey-radio-option--active' : ''}`}
            onClick={() => onChange(option.value)}
          >
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={isSelected}
              onChange={(e) => onChange(e.target.value)}
              className="survey-radio-input"
              readOnly
            />
            <div className="survey-radio-label">
              {showEmoji && option.emoji && <span className="survey-radio-emoji">{option.emoji}</span>}
              <span className="survey-radio-text">{option.label}</span>
              {option.desc && <span className="survey-radio-desc">{option.desc}</span>}
            </div>
          </div>
        );
      })}
    </div>
  );

  return (
    <div className="background-survey">
      {/* Programming Experience Section */}
      <div className="survey-section">
        <h3 className="survey-section-title">
          <span className="survey-section-icon">💻</span>
          Programming Experience
        </h3>
        <p className="survey-section-desc">How comfortable are you with these languages?</p>

        <div className="survey-field">
          <label className="survey-label">Python</label>
          <RadioGroup
            name="python_experience"
            value={formData.python_experience}
            options={experienceLevels}
            onChange={(value) => handleChange('python_experience', value)}
            showEmoji
          />
        </div>

        <div className="survey-field">
          <label className="survey-label">C++</label>
          <RadioGroup
            name="cpp_experience"
            value={formData.cpp_experience}
            options={experienceLevels}
            onChange={(value) => handleChange('cpp_experience', value)}
            showEmoji
          />
        </div>

        <div className="survey-field">
          <label className="survey-label">JavaScript/TypeScript</label>
          <RadioGroup
            name="js_ts_experience"
            value={formData.js_ts_experience}
            options={experienceLevels}
            onChange={(value) => handleChange('js_ts_experience', value)}
            showEmoji
          />
        </div>
      </div>

      {/* Robotics & AI Section */}
      <div className="survey-section">
        <h3 className="survey-section-title">
          <span className="survey-section-icon">🤖</span>
          Robotics & AI Background
        </h3>
        <p className="survey-section-desc">Your experience with AI/ML and robotics frameworks</p>

        <div className="survey-field">
          <label className="survey-label">AI/ML Familiarity</label>
          <RadioGroup
            name="ai_ml_familiarity"
            value={formData.ai_ml_familiarity}
            options={experienceLevels}
            onChange={(value) => handleChange('ai_ml_familiarity', value)}
            showEmoji
          />
        </div>

        <div className="survey-field">
          <label className="survey-label">ROS 2 Experience</label>
          <RadioGroup
            name="ros2_experience"
            value={formData.ros2_experience}
            options={experienceLevels}
            onChange={(value) => handleChange('ros2_experience', value)}
            showEmoji
          />
        </div>
      </div>

      {/* Hardware Setup Section */}
      <div className="survey-section">
        <h3 className="survey-section-title">
          <span className="survey-section-icon">⚙️</span>
          Hardware Setup
        </h3>
        <p className="survey-section-desc">Your system specifications</p>

        <div className="survey-field">
          <label className="survey-label">GPU</label>
          <RadioGroup
            name="gpu_details"
            value={formData.gpu_details}
            options={gpuOptions}
            onChange={(value) => handleChange('gpu_details', value)}
          />
        </div>

        <div className="survey-field">
          <label className="survey-label">RAM</label>
          <RadioGroup
            name="ram_capacity"
            value={formData.ram_capacity}
            options={ramOptions}
            onChange={(value) => handleChange('ram_capacity', value)}
          />
        </div>

        <div className="survey-field">
          <label className="survey-label">Operating System</label>
          <RadioGroup
            name="operating_system"
            value={formData.operating_system}
            options={osOptions}
            onChange={(value) => handleChange('operating_system', value)}
            showEmoji
          />
        </div>
      </div>

      {/* Additional Hardware Section */}
      <div className="survey-section">
        <h3 className="survey-section-title">
          <span className="survey-section-icon">🎯</span>
          Additional Hardware
        </h3>
        <p className="survey-section-desc">Optional hardware for hands-on projects</p>

        <div className="survey-checkbox-group">
          <label className="survey-checkbox-label">
            <input
              type="checkbox"
              checked={formData.jetson_ownership}
              onChange={(e) => handleCheckboxChange('jetson_ownership', e.target.checked)}
              className="survey-checkbox-input"
            />
            <span className="survey-checkbox-text">
              <span className="survey-checkbox-title">NVIDIA Jetson</span>
              <span className="survey-checkbox-desc">I own Jetson Nano, Xavier, or Orin</span>
            </span>
          </label>

          <label className="survey-checkbox-label">
            <input
              type="checkbox"
              checked={formData.realsense_lidar_availability}
              onChange={(e) => handleCheckboxChange('realsense_lidar_availability', e.target.checked)}
              className="survey-checkbox-input"
            />
            <span className="survey-checkbox-text">
              <span className="survey-checkbox-title">Sensors</span>
              <span className="survey-checkbox-desc">I have RealSense camera or LiDAR</span>
            </span>
          </label>
        </div>
      </div>

      {/* Info Box */}
      <div className="survey-info-box">
        <div className="survey-info-icon">💡</div>
        <div>
          <p className="survey-info-title">Why we ask</p>
          <p className="survey-info-text">
            We use this information to personalize content difficulty, provide relevant examples, 
            and suggest alternative approaches based on your hardware capabilities.
          </p>
        </div>
      </div>
    </div>
  );
};

export default BackgroundSurvey;