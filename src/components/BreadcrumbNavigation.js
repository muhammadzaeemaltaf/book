import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useLocation } from '@docusaurus/router';

/**
 * BreadcrumbNavigation Component
 * Provides breadcrumb navigation for textbook content
 */
export default function BreadcrumbNavigation({ items = [] }) {
  const { siteConfig } = useDocusaurusContext();
  const location = useLocation();

  // If items are not provided, we can generate them based on the current path
  // For now, we'll use the provided items
  if (!items.length) {
    return null;
  }

  return (
    <nav className="breadcrumb-navigation" aria-label="Breadcrumb">
      <ol className="breadcrumb-navigation__list">
        <li className="breadcrumb-navigation__item">
          <Link to="/" className="breadcrumb-navigation__link">
            Home
          </Link>
        </li>
        {items.map((item, index) => (
          <li key={index} className="breadcrumb-navigation__separator">
            <span aria-hidden="true">›</span>
          </li>
        ))}
        {items.map((item, index) => (
          <li key={`item-${index}`} className="breadcrumb-navigation__item">
            {item.url ? (
              <Link to={item.url} className="breadcrumb-navigation__link">
                {item.title}
              </Link>
            ) : (
              <span className="breadcrumb-navigation__current">
                {item.title}
              </span>
            )}
          </li>
        ))}
      </ol>

      <style jsx>{`
        .breadcrumb-navigation {
          margin: 1rem 0;
          padding: 0.75rem 0;
        }

        .breadcrumb-navigation__list {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          list-style: none;
          margin: 0;
          padding: 0;
        }

        .breadcrumb-navigation__item {
          display: inline-block;
        }

        .breadcrumb-navigation__item:last-child .breadcrumb-navigation__current {
          color: #6b7280;
          font-weight: 500;
        }

        .breadcrumb-navigation__link {
          color: #374151;
          text-decoration: none;
          font-size: 0.875rem;
        }

        .breadcrumb-navigation__link:hover {
          text-decoration: underline;
        }

        .breadcrumb-navigation__current {
          color: #6b7280;
          font-size: 0.875rem;
        }

        .breadcrumb-navigation__separator {
          margin: 0 0.5rem;
          color: #9ca3af;
        }
      `}</style>
    </nav>
  );
}