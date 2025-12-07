import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: '🤖 ROS 2 Mastery',
    description: (
      <>
        Deep dive into ROS 2 architecture, nodes, topics, services, and practical implementations
        for real-world robotics applications.
      </>
    ),
    to: '/docs/module-01-ros2/chapter-01-01-architecture',
  },
  {
    title: '🎮 Digital Twin Simulation',
    description: (
      <>
        Master Gazebo, Unity, and NVIDIA Isaac simulation environments for testing
        and validating your robotics algorithms.
      </>
    ),
    to: '/docs/module-02-digital-twin/chapter-02-01-simulation-fundamentals',
  },
  {
    title: '🧠 Vision-Language-Action Systems',
    description: (
      <>
        Learn how to integrate VLA models with physical robots for advanced
        perception and manipulation capabilities.
      </>
    ),
    to: '/docs/module-04-vla/chapter-04-01-vla-fundamentals',
  },
];

function Feature({title, description, to}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
        <Link className="button button--secondary button--sm" to={to}>
          Learn More
        </Link>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}