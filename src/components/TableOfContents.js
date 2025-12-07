import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

/**
 * TableOfContents Component
 * Displays the complete textbook structure
 */
export default function TableOfContents({ modules = [] }) {
  const { siteConfig } = useDocusaurusContext();

  if (!modules.length) {
    return null;
  }

  return (
    <div className="table-of-contents">
      <div className="table-of-contents__header">
        <h2>Table of Contents</h2>
        <p className="table-of-contents__subtitle">Complete Physical AI & Humanoid Robotics Textbook Structure</p>
      </div>

      <div className="table-of-contents__modules">
        {modules.map((module, moduleIndex) => (
          <div key={moduleIndex} className="table-of-contents__module">
            <h3 className="table-of-contents__module-title">
              {module.title}
            </h3>
            {module.description && (
              <p className="table-of-contents__module-description">{module.description}</p>
            )}
            <ul className="table-of-contents__chapters">
              {module.chapters && module.chapters.map((chapter, chapterIndex) => (
                <li key={chapterIndex} className="table-of-contents__chapter">
                  <Link to={chapter.url} className="table-of-contents__chapter-link">
                    {chapter.title}
                  </Link>
                  {chapter.description && (
                    <p className="table-of-contents__chapter-description">{chapter.description}</p>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <style jsx>{`
        .table-of-contents {
          margin: 2rem 0;
          padding: 1rem;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          // background-color: #fafafa;
        }

        .table-of-contents__header {
          margin-bottom: 2rem;
          text-align: center;
        }

        .table-of-contents__header h2 {
          margin: 0 0 0.5rem 0;
          color: #1f2937;
          font-size: 1.875rem;
        }

        .table-of-contents__subtitle {
          margin: 0;
          color: #6b7280;
          font-size: 1rem;
        }

        .table-of-contents__modules {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .table-of-contents__module {
          padding: 1rem;
          border: 1px solid #e5e7eb;
          border-radius: 6px;
          background-color: white;
        }

        .table-of-contents__module-title {
          margin: 0 0 0.75rem 0;
          color: #374151;
          font-size: 1.25rem;
        }

        .table-of-contents__module-description {
          margin: 0 0 1rem 0;
          color: #6b7280;
          font-size: 0.875rem;
        }

        .table-of-contents__chapters {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .table-of-contents__chapter {
          margin-bottom: 0.5rem;
          padding-left: 1rem;
          position: relative;
        }

        .table-of-contents__chapter::before {
          content: "•";
          position: absolute;
          left: 0;
          color: #3b82f6;
          font-weight: bold;
        }

        .table-of-contents__chapter-link {
          font-weight: 500;
          color: #1d4ed8;
          text-decoration: none;
          font-size: 0.95rem;
        }

        .table-of-contents__chapter-link:hover {
          text-decoration: underline;
        }

        .table-of-contents__chapter-description {
          margin: 0.25rem 0 0 0;
          color: #6b7280;
          font-size: 0.8125rem;
          padding-left: 1rem;
        }
      `}</style>
    </div>
  );
}