import React, {type ReactNode} from 'react';
import Content from '@theme-original/Navbar/Content';
import type ContentType from '@theme/Navbar/Content';
import type {WrapperProps} from '@docusaurus/types';
import { useAuth } from '../../../contexts/AuthContext';
import UserAvatar from '../../../components/UserAvatar';
import styles from '../navbar.module.css';

// @ts-ignore
import { UserDropdown } from '../AuthComponents';

type Props = WrapperProps<typeof ContentType>;

export default function ContentWrapper(props: Props): ReactNode {
  const { isAuthenticated, user, loading, signout } = useAuth();

  return (
    <>
      <Content {...props} />
      <div className={styles.authNavbarItem}>
        {loading ? (
          <div className={styles.loadingSpinner} />
        ) : isAuthenticated && user ? (
          <UserDropdown user={user} onSignOut={signout} />
        ) : (
          <div className={styles.authButtons}>
            <a href="/signin" className="button button--secondary button--sm">
              Sign In
            </a>
            <a href="/signup" className="button button--primary button--sm">
              Sign Up
            </a>
          </div>
        )}
      </div>
    </>
  );
}
