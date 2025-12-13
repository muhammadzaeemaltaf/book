import React, { useState, useEffect, useRef } from 'react';
import UserAvatar from '../../components/UserAvatar';
import styles from './navbar.module.css';

// User dropdown menu component
export const UserDropdown = ({ user, onSignOut }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const handleSignOut = async () => {
    setIsOpen(false);
    if (onSignOut) {
      await onSignOut();
    }
    window.location.href = '/';
  };

  return (
    <div className={styles.userDropdown} ref={dropdownRef}>
      <div
        className={styles.userDropdownTrigger}
        onClick={() => setIsOpen(!isOpen)}
      >
        <UserAvatar user={user} size={36} showName={true} />
        <svg
          className={`${styles.chevron} ${isOpen ? styles.chevronOpen : ''}`}
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="currentColor"
        >
          <path d="M4.427 6.427a.75.75 0 011.06 0L8 8.94l2.513-2.513a.75.75 0 111.06 1.06l-3 3a.75.75 0 01-1.06 0l-3-3a.75.75 0 010-1.06z" />
        </svg>
      </div>

      {isOpen && (
        <div className={styles.userDropdownMenu}>
          <div className={styles.userDropdownHeader}>
            <div className={styles.userName}>{user.name || 'User'}</div>
            <div className={styles.userEmail}>{user.email}</div>
          </div>
          <div className={styles.userDropdownDivider} />
          <a href="/profile" className={styles.userDropdownItem}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 8a3 3 0 100-6 3 3 0 000 6zm2 3H6a4 4 0 00-4 4v1h12v-1a4 4 0 00-4-4z" />
            </svg>
            Profile Settings
          </a>
          <a href="/dashboard" className={styles.userDropdownItem}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M2 4a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V4zm2-1a1 1 0 00-1 1v1h10V4a1 1 0 00-1-1H4zm9 4H3v5a1 1 0 001 1h8a1 1 0 001-1V7z" />
            </svg>
            Dashboard
          </a>
          <div className={styles.userDropdownDivider} />
          <button
            onClick={handleSignOut}
            className={`${styles.userDropdownItem} ${styles.userDropdownItemDanger}`}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M5 3.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm0 2.122a2.25 2.25 0 10-1.5 0v.878A2.25 2.25 0 005.75 8.5h1.5v2.128a2.251 2.251 0 101.5 0V8.5h1.5a2.25 2.25 0 002.25-2.25v-.878a2.25 2.25 0 10-1.5 0v.878a.75.75 0 01-.75.75h-4.5A.75.75 0 015 6.25v-.878zm3.75 7.378a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm3-8.75a.75.75 0 100-1.5.75.75 0 000 1.5z" />
            </svg>
            Sign Out
          </button>
        </div>
      )}
    </div>
  );
};
