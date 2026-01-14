# Seer Documentation

This directory contains the Docusaurus-based documentation for Seer, deployed at [docs.getseer.dev](https://docs.getseer.dev).

## Features

✅ **Docusaurus 3** - Modern static site generator
✅ **Copyable Code Snippets** - Built-in copy buttons on all code blocks
✅ **AI-Readable** - llms.txt and llms-full.txt for Claude, ChatGPT, Cursor, Windsurf
✅ **Search Ready** - Configured for Algolia DocSearch (free)
✅ **Dark Mode** - Auto-detects system preference
✅ **Mobile Responsive** - Works great on all devices
✅ **Zero Cost Deployment** - Free hosting on Vercel/Netlify/GitHub Pages

## Quick Start

### Development

```bash
npm install
npm start
```

Opens http://localhost:3000 with live reload.

### Build

```bash
npm run build
```

### Test Production Build

```bash
npm run serve
```

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.

**Quick Deploy Options:**
- Vercel: `vercel --prod`
- Netlify: `netlify deploy --prod`
- GitHub Pages: Automatic via GitHub Actions

## Resources

- [Docusaurus Documentation](https://docusaurus.io/)
- [Algolia DocSearch](https://docsearch.algolia.com/)
- [llms.txt Standard](https://llms-txt.io/)
