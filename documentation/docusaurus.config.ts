import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Seer Documentation',
  tagline: 'Workflow builder with fine-grained control for automated workflows',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://docs.getseer.dev',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'seer-engg', // Usually your GitHub org/user name.
  projectName: 'seer', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  plugins: [
    [
      '@docusaurus/plugin-google-analytics',
      {
        trackingID: process.env.GA_MEASUREMENT_ID,
        anonymizeIP: true,
      },
    ],
  ],

  clientModules: [
    require.resolve('./src/clientModules/analytics.ts'),
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/seer-engg/seer/tree/main/documentation/',
          routeBasePath: '/', // Serve docs at the root
        },
        blog: false, // Disable blog
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    // Algolia DocSearch configuration
    algolia: {
      appId: '4FRO5P9OUZ',
      apiKey: 'ff2707a0fe9916f1d9f88fc855c45263',
      indexName: 'Docs crawler',
      contextualSearch: true,
      searchParameters: {},
      insights: true, // Enable search analytics
      askAi: 'wr254tv8zZQO',
    },
    navbar: {
      title: 'Seer',
      logo: {
        alt: 'Seer Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://github.com/seer-engg/seer',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Quick Start',
              to: '/',
            },
            {
              label: 'Configuration',
              to: '/advanced/CONFIGURATION',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/seer-engg/seer',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Seer. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'docker', 'typescript', 'javascript'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
