import React from 'react';

const SignupFeedback = ({ type, message }) => {
  const getFeedbackStyle = () => {
    switch (type) {
      case 'success':
        return {
          container: 'callout callout-tip',
          icon: '✅'
        };
      case 'error':
        return {
          container: 'callout callout-error',
          icon: '❌'
        };
      case 'warning':
        return {
          container: 'callout callout-warning',
          icon: '⚠️'
        };
      case 'info':
      default:
        return {
          container: 'callout callout-info',
          icon: 'ℹ️'
        };
    }
  };

  const { container, icon } = getFeedbackStyle();

  return (
    <div className={`rounded-md p-4 mb-4 ${container}`} role={type === 'error' ? 'alert' : 'status'} aria-live="polite">
      <div className="flex">
        <div className="flex-shrink-0 text-xl mr-3">{icon}</div>
        <div className="flex-1">
          <p className="text-sm">{message}</p>
        </div>
      </div>
    </div>
  );
};

export default SignupFeedback;