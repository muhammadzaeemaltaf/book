import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import { useAuth } from '../contexts/AuthContext';
import { useHistory } from '@docusaurus/router';

const DashboardPage = () => {
  const { isAuthenticated, user, loading } = useAuth();
  const history = useHistory();

  useEffect(() => {
    // Redirect to signin if not authenticated
    if (!loading && !isAuthenticated) {
      const currentPath = window.location.pathname;
      history.push(`/signin?return=${encodeURIComponent(currentPath)}`);
    }
  }, [isAuthenticated, loading, history]);

  if (loading) {
    return (
      <Layout title="Dashboard" description="User Dashboard">
        <div className="container margin-vert--lg">
          <div className="text-center">
            <p>Loading...</p>
          </div>
        </div>
      </Layout>
    );
  }

  if (!isAuthenticated) {
    return null; // Will redirect
  }

  return (
    <Layout title="Dashboard" description="User Dashboard">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col">
            <h1>Welcome to Your Dashboard</h1>
            <p>Hello, {user?.name || user?.email}!</p>
            
            <div className="row margin-top--lg">
              <div className="col col--4">
                <div className="card">
                  <div className="card__header">
                    <h3>Quick Stats</h3>
                  </div>
                  <div className="card__body">
                    <p>Your learning progress and statistics will appear here.</p>
                  </div>
                </div>
              </div>
              
              <div className="col col--4">
                <div className="card">
                  <div className="card__header">
                    <h3>Recent Activity</h3>
                  </div>
                  <div className="card__body">
                    <p>Your recent activities and interactions will be shown here.</p>
                  </div>
                </div>
              </div>
              
              <div className="col col--4">
                <div className="card">
                  <div className="card__header">
                    <h3>Quick Actions</h3>
                  </div>
                  <div className="card__body">
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <a href="/profile" className="button button--primary button--block">
                        Edit Profile
                      </a>
                      <a href="/docs/intro" className="button button--secondary button--block">
                        Continue Learning
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default DashboardPage;
