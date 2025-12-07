import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="row">
          {/* Hero Content (Centered) */}
          <div className="col col--12">
            <div className={clsx('hero-content', styles.heroContent)}>
              {/* Animated Badge */}
              <div className={clsx('hero-badge', 'animate-slide-down', styles.heroBadge)}>
                <span>🤖 Advanced Robotics Curriculum</span>
              </div>

              {/* Main Title - Animated */}
              <Heading as="h1" className={clsx('hero-title', 'animate-fade-in-up', 'delay-100')}>
                {siteConfig.title}
              </Heading>

              {/* Subtitle - Animated */}
              <Heading as="h2" className={clsx('hero-subtitle', 'animate-fade-in-up', 'delay-200')}>
                Building Intelligent Humanoid Robots
              </Heading>

              {/* Description - Animated */}
              <p className={clsx('hero-description', 'animate-fade-in-up', 'delay-300')}>
                Master ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems.
                Transform your AI knowledge into physical robotics expertise through
                hands-on projects and real-world simulations.
              </p>

              {/* CTA Buttons - Animated */}
              <div className={clsx('hero-cta-group', 'animate-fade-in-up', 'delay-400')}>
                <Link
                  className="button button--secondary button--lg btn-primary"
                  style={{ marginRight: '12px' }}
                  to="/docs/intro">
                  Start Learning 🚀
                </Link>
                <Link
                  className="button button--outline button--lg btn-secondary"
                  to="/docs/table-of-contents">
                  View Syllabus
                </Link>
              </div>

              {/* Key Features - Animated */}
              <div className={clsx('hero-features', 'animate-fade-in-up', 'delay-500')} style={{ marginTop: '24px', textAlign: 'left' }}>
                <div className="feature-item">
                  <span>✅</span>
                  <span>13-Week Structured Curriculum</span>
                </div>
                <div className="feature-item">
                  <span>✅</span>
                  <span>50+ Hands-on Code Examples</span>
                </div>
                <div className="feature-item">
                  <span>✅</span>
                  <span>Real-World Capstone Project</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Animated Stats Row */}
        <div className={clsx('row', 'stats-row', styles.statsRow)}>
          <div className={clsx('col', 'col--3', 'animate-scale-in', 'delay-600')}>
            <div className="stat-card">
              <div className="stat-number" style={{fontSize: '28px', fontWeight: '700', color: 'var(--color-primary)', lineHeight: '1', marginBottom: '8px'}}>4</div>
              <div className="stat-label">Core Modules</div>
            </div>
          </div>
          <div className={clsx('col', 'col--3', 'animate-scale-in', 'delay-700')}>
            <div className="stat-card">
              <div className="stat-number" style={{fontSize: '28px', fontWeight: '700', color: 'var(--color-primary)', lineHeight: '1', marginBottom: '8px'}}>50+</div>
              <div className="stat-label">Code Examples</div>
            </div>
          </div>
          <div className={clsx('col', 'col--3', 'animate-scale-in', 'delay-800')}>
            <div className="stat-card">
              <div className="stat-number" style={{fontSize: '28px', fontWeight: '700', color: 'var(--color-primary)', lineHeight: '1', marginBottom: '8px'}}>13</div>
              <div className="stat-label">Weeks</div>
            </div>
          </div>
          <div className={clsx('col', 'col--3', 'animate-scale-in', 'delay-900')}>
            <div className="stat-card">
              <div className="stat-number" style={{fontSize: '28px', fontWeight: '700', color: 'var(--color-primary)', lineHeight: '1', marginBottom: '8px'}}>100%</div>
              <div className="stat-label">Hands-on</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Bridging digital AI knowledge to physical robotics for intermediate AI/software developers">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}