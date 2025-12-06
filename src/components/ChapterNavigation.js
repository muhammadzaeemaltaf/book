import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

/**
 * ChapterNavigation Component
 * Provides next and previous chapter navigation links
 */
export default function ChapterNavigation({ previous, next }) {
  const { siteConfig } = useDocusaurusContext();

  if (!previous && !next) {
    return null;
  }

  return (
    <div className="chapter-navigation">
      <div className="chapter-navigation__container">
        {previous && (
          <div className="chapter-navigation__previous">
            <h4>Previous</h4>
            <Link to={previous.permalink} className="chapter-navigation__link">
              ← {previous.title}
            </Link>
          </div>
        )}

        {next && (
          <div className="chapter-navigation__next">
            <h4>Next</h4>
            <Link to={next.permalink} className="chapter-navigation__link">
              {next.title} →
            </Link>
          </div>
        )}
      </div>

      <style jsx>{`
        .chapter-navigation {
          margin: 2rem 0;
          padding: 1.5rem 0;
          border-top: 1px solid #e5e7eb;
          border-bottom: 1px solid #e5e7eb;
        }

        .chapter-navigation__container {
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .chapter-navigation__previous,
        .chapter-navigation__next {
          flex: 1;
          min-width: 200px;
        }

        .chapter-navigation__previous h4,
        .chapter-navigation__next h4 {
          margin: 0 0 0.5rem 0;
          font-size: 0.875rem;
          font-weight: 600;
          color: #6b7280;
          text-transform: uppercase;
        }

        .chapter-navigation__link {
          font-weight: 500;
          color: #1d4ed8;
          text-decoration: none;
          font-size: 1rem;
        }

        .chapter-navigation__link:hover {
          text-decoration: underline;
        }

        @media (max-width: 768px) {
          .chapter-navigation__container {
            flex-direction: column;
            align-items: stretch;
          }

          .chapter-navigation__previous,
          .chapter-navigation__next {
            min-width: auto;
          }
        }
      `}</style>
    </div>
  );
}