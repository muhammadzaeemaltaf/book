import { themes as prismThemes } from "prism-react-renderer";

const isGithubPages = process.env.DOCUSAURUS_ENV === 'github';

/** @type {import('@docusaurus/types').Config} */

const config = {
  title: "Physical AI & Humanoid Robotics Textbook",
  tagline:
    "Bridging digital AI knowledge to physical robotics for intermediate AI/software developers",
  favicon: "img/favicon.ico",

  // Set the production url of your site here
  url: "https://humanoid-robotics-textbook-eight.vercel.app/",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub Pages deployment, it's usually '/<orgName>/<repoName>/'
  baseUrl: isGithubPages ? "/book/" : "/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "muhammadzaeemaltaf", // Usually your GitHub org/user name.
  projectName: "book", // Usually your repo name.
  trailingSlash: false,
  deploymentBranch: "gh-pages", // Branch that GitHub Pages will deploy from

  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",
  markdown: {
    mermaid: true,
    mdx1Compat: {
      comments: true,
      admonitions: true,
      headingIds: true,
    },
  },

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: "./sidebars.js",
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl: "https://github.com/muhammadzaeemaltaf/book/tree/main/",
          breadcrumbs: true, // Enable breadcrumbs
          routeBasePath: "/docs", // Ensure docs are served from /docs/ path
        },
        blog: false, // Disable blog for textbook
        theme: {
          customCss: "./src/css/custom.css",
        },
      }),
    ],
  ],

  plugins: [
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        hashed: true,
        indexDocs: true,
        indexBlog: false,
        indexPages: false,
        language: ["en"],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        docsRouteBasePath: "/docs",
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: "img/docusaurus-social-card.jpg",
      navbar: {
        title: "Physical AI & Robotics Textbook",
        items: [
          {
            type: "doc",
            docId: "intro",
            position: "left",
            label: "Textbook",
          },
          {
            href: "https://github.com/muhammadzaeemaltaf/book",
            label: "GitHub",
            position: "right",
          },
          {
            type: "search",
            position: "right",
          },
        ],
      },
      docs: {
        sidebar: {
          hideable: true,
        },
      },
      theme: {
        customCss: [
          "./src/css/custom.css",
          "./src/css/sidebar.css",
          "./src/css/doc-pages.css",
          "./src/css/chat.css", // Add chat CSS
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Textbook",
            items: [
              {
                label: "Introduction",
                to: "/docs/intro",
              },
            ],
          },
          {
            title: "Community",
            items: [
              {
                label: "Stack Overflow",
                href: "https://stackoverflow.com/questions/tagged/docusaurus",
              },
              {
                label: "Discord",
                href: "https://discordapp.com/invite/docusaurus",
              },
            ],
          },
          {
            title: "More",
            items: [
              {
                label: "GitHub",
                href: "https://github.com/muhammadzaeemaltaf/book",
              },
              {
                label: "LinkedIn",
                href: "https://www.linkedin.com/in/muhammadzaeemaltaf/",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
      markdown: {
        mermaid: true,
        mdx1Compat: {
          comments: true,
          admonitions: true,
          headingIds: true,
        },
      },
    }),
};

export default config;
