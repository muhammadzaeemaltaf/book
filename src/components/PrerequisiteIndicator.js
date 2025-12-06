import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

/**
 * PrerequisiteIndicator Component
 * Shows prerequisites for a chapter/module
 */
export default function PrerequisiteIndicator({ prerequisites = [], title = "Prerequisites" }) {
  const { siteConfig } = useDocusaurusContext();

  if (!prerequisites.length) {
    return null;
  }

  return (
    <div className="prerequisite-indicator">
      <div className="prerequisite-indicator__header">
        <h3>{title}</h3>
      </div>
      <ul className="prerequisite-indicator__list">
        {prerequisites.map((prereq, index) => (
          <li key={index} className="prerequisite-indicator__item">
            {prereq.url ? (
              <Link to={prereq.url} className="prerequisite-indicator__link">
                {prereq.title || prereq}
              </Link>
            ) : (
              <span className="prerequisite-indicator__text">
                {prereq.title || prereq}
              </span>
            )}
            {prereq.description && (
              <p className="prerequisite-indicator__description">{prereq.description}</p>
            )}
          </li>
        ))}
      </ul>

      <style jsx>{`
        .prerequisite-indicator {
          margin: 1.5rem 0;
          padding: 1.25rem;
          border: 2px solid #dbeafe;
          border-radius: 8px;
          background-color: #eff6ff;
        }

        .prerequisite-indicator__header h3 {
          margin: 0 0 1rem 0;
          color: #1d428a;
          font-size: 1.125rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .prerequisite-indicator__list {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .prerequisite-indicator__item {
          margin-bottom: 0.75rem;
          padding-left: 1.5rem;
          position: relative;
        }

        .prerequisite-indicator__item::before {
          content: "•";
          position: absolute;
          left: 0;
          color: #3b82f6;
          font-weight: bold;
        }

        .prerequisite-indicator__link {
          font-weight: 500;
          color: #1d4ed8;
          text-decoration: none;
        }

        .prerequisite-indicator__link:hover {
          text-decoration: underline;
        }

        .prerequisite-indicator__text {
          font-weight: 500;
          color: #374151;
        }

        .prerequisite-indicator__description {
          margin: 0.25rem 0 0 0;
          color: #6b7280;
          font-size: 0.875rem;
        }
      `}</style>
    </div>
  );
}