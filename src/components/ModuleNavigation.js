import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';

/**
 * ModuleNavigation Component
 * Provides navigation links between related modules and chapters
 */
export default function ModuleNavigation({ currentModule, relatedModules = [], relatedChapters = [] }) {
  const { siteConfig } = useDocusaurusContext();

  if (!relatedModules.length && !relatedChapters.length) {
    return null;
  }

  return (
    <div className="module-navigation">
      <div className="module-navigation__header">
        <h3>Cross-Module References</h3>
      </div>

      {relatedModules.length > 0 && (
        <div className="module-navigation__section">
          <h4>Related Modules</h4>
          <ul className="module-navigation__list">
            {relatedModules.map((module, index) => (
              <li key={index} className="module-navigation__item">
                <Link to={useBaseUrl(module.url)}>
                  {module.title}
                </Link>
                {module.description && (
                  <p className="module-navigation__description">{module.description}</p>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {relatedChapters.length > 0 && (
        <div className="module-navigation__section">
          <h4>Related Chapters</h4>
          <ul className="module-navigation__list">
            {relatedChapters.map((chapter, index) => (
              <li key={index} className="module-navigation__item">
                <Link to={useBaseUrl(chapter.url)}>
                  {chapter.title}
                </Link>
                {chapter.description && (
                  <p className="module-navigation__description">{chapter.description}</p>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      <style jsx>{`
        .module-navigation {
          margin: 2rem 0;
          padding: 1.5rem;
          border: 1px solid #e0e0e0;
          border-radius: 8px;
          background-color: #f9f9f9;
        }

        .module-navigation__header h3 {
          margin: 0 0 1rem 0;
          color: #2563eb;
          font-size: 1.25rem;
        }

        .module-navigation__section {
          margin-bottom: 1.5rem;
        }

        .module-navigation__section h4 {
          margin: 0 0 0.75rem 0;
          color: #374151;
          font-size: 1rem;
          font-weight: 600;
        }

        .module-navigation__list {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .module-navigation__item {
          margin-bottom: 0.5rem;
        }

        .module-navigation__item a {
          font-weight: 500;
          color: #1d4ed8;
          text-decoration: none;
        }

        .module-navigation__item a:hover {
          text-decoration: underline;
        }

        .module-navigation__description {
          margin: 0.25rem 0 0 0;
          color: #6b7280;
          font-size: 0.875rem;
        }
      `}</style>
    </div>
  );
}