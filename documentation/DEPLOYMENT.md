# Documentation Deployment Guide

This guide explains how to deploy your Seer documentation to various hosting platforms.

## Prerequisites

- Documentation built successfully (`npm run build`)
- GitHub repository with the documentation code
- DNS configured for docs.getseer.dev (if using custom domain)

## Option 1: Vercel (Recommended)

Vercel offers the easiest deployment experience with automatic deployments on git push.

### Steps:

1. **Install Vercel CLI** (optional):
   ```bash
   npm i -g vercel
   ```

2. **Deploy via Vercel Dashboard**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Set **Root Directory** to `documentation`
   - Framework Preset: Other
   - Build Command: `npm run build`
   - Output Directory: `build`
   - Click "Deploy"

3. **Configure Custom Domain**:
   - Go to Project Settings → Domains
   - Add `docs.getseer.dev`
   - Follow DNS configuration instructions
   - Vercel handles SSL automatically

4. **Environment Variables** (if needed):
   - Go to Project Settings → Environment Variables
   - Add any required variables

### CLI Deployment:

```bash
cd documentation
vercel --prod
```

**Estimated Cost:** Free for open source projects

---

## Option 2: Netlify

Netlify is another excellent option with similar features to Vercel.

### Steps:

1. **Deploy via Netlify Dashboard**:
   - Go to [netlify.com](https://netlify.com)
   - Click "Add new site" → "Import an existing project"
   - Connect to GitHub and select your repository
   - Configure build settings:
     - Base directory: `documentation`
     - Build command: `npm run build`
     - Publish directory: `documentation/build`
   - Click "Deploy site"

2. **Configure Custom Domain**:
   - Go to Site Settings → Domain management
   - Add custom domain: `docs.getseer.dev`
   - Follow DNS configuration instructions
   - Netlify handles SSL automatically

3. **Continuous Deployment**:
   - Netlify automatically deploys on every push to main branch
   - Configure via `netlify.toml` (already included)

### CLI Deployment:

```bash
npm install -g netlify-cli
cd documentation
netlify deploy --prod
```

**Estimated Cost:** Free for open source projects

---

## Option 3: GitHub Pages

GitHub Pages is free for public repositories and integrates seamlessly with GitHub Actions.

### Steps:

1. **Enable GitHub Pages**:
   - Go to your repository settings
   - Navigate to "Pages"
   - Source: "GitHub Actions"

2. **GitHub Actions Workflow**:
   - The workflow is already configured in `.github/workflows/deploy-docs.yml`
   - Pushes to `main` branch automatically trigger deployment
   - Workflow runs in the `documentation` directory

3. **Configure Custom Domain**:
   - Go to Repository Settings → Pages
   - Add custom domain: `docs.getseer.dev`
   - Create CNAME record in your DNS:
     ```
     CNAME docs.getseer.dev -> your-username.github.io
     ```
   - GitHub handles SSL automatically

4. **Manual Deployment** (if needed):
   ```bash
   cd documentation
   npm run build

   # Use gh-pages for manual deployment
   npm install -g gh-pages
   gh-pages -d build
   ```

**Estimated Cost:** Free for public repositories

---

## Option 4: Railway

If you're already using Railway for your backend, you can deploy docs there too.

### Steps:

1. **Create New Service**:
   - Go to [railway.app](https://railway.app)
   - Select your project
   - Click "New" → "GitHub Repo"
   - Select your repository

2. **Configure Service**:
   - Root directory: `documentation`
   - Build command: `npm run build`
   - Start command: `npx serve -s build -l $PORT`

3. **Add Build Dependencies**:
   ```bash
   cd documentation
   npm install --save-dev serve
   ```

4. **Configure Domain**:
   - Go to service settings → Networking
   - Add custom domain: `docs.getseer.dev`

**Estimated Cost:** ~$5/month

---

## Post-Deployment: Algolia DocSearch

Once your documentation is live, apply for free Algolia DocSearch:

1. **Apply for DocSearch**:
   - Go to [docsearch.algolia.com/apply](https://docsearch.algolia.com/apply/)
   - Fill out the form with:
     - Documentation URL: `https://docs.getseer.dev`
     - Email: your@email.com
     - Repository: https://github.com/seer-engg/seer

2. **Configure DocSearch**:
   - Algolia will email you API keys
   - Update `documentation/docusaurus.config.ts`:
     ```typescript
     algolia: {
       appId: 'YOUR_APP_ID',
       apiKey: 'YOUR_SEARCH_API_KEY',
       indexName: 'seer',
       contextualSearch: true,
     },
     ```

3. **Rebuild and Deploy**:
   ```bash
   npm run build
   # Deploy using your chosen platform
   ```

**Approval Time:** Usually 1-3 business days

---

## Testing Your Deployment

After deployment, verify everything works:

1. **Homepage loads**: https://docs.getseer.dev
2. **Navigation works**: Click through different sections
3. **Code snippets have copy buttons**: Test on any code block
4. **llms.txt is accessible**: https://docs.getseer.dev/llms.txt
5. **llms-full.txt is accessible**: https://docs.getseer.dev/llms-full.txt
6. **Search works** (after Algolia is configured)

## Troubleshooting

### Build Fails

```bash
# Clear cache and rebuild
cd documentation
rm -rf node_modules .docusaurus
npm install
npm run build
```

### Custom Domain Not Working

1. Verify DNS records are propagated: `dig docs.getseer.dev`
2. Check SSL certificate status in hosting dashboard
3. Wait 24-48 hours for DNS propagation
4. Ensure CNAME record points to correct host

### llms.txt Files Not Accessible

- Verify files exist in `documentation/static/`
- Check hosting platform serves static files correctly
- Verify Content-Type header is set to `text/plain`

## Recommended Setup

For production, we recommend:

1. **Hosting**: Vercel or Netlify (both are free and excellent)
2. **DNS**: Cloudflare for additional features (optional)
3. **Search**: Algolia DocSearch (free for docs sites)
4. **Monitoring**: UptimeRobot or similar for uptime monitoring

## Continuous Deployment

All platforms support automatic deployments:

- **Vercel/Netlify**: Automatically deploy on every push to main
- **GitHub Pages**: Uses GitHub Actions workflow (included)
- **Railway**: Auto-deploys on git push

## Cost Summary

| Platform | Monthly Cost | Custom Domain | SSL | Auto Deploy |
|----------|-------------|---------------|-----|-------------|
| Vercel | Free | Yes | Yes | Yes |
| Netlify | Free | Yes | Yes | Yes |
| GitHub Pages | Free | Yes | Yes | Yes |
| Railway | ~$5 | Yes | Yes | Yes |

**Recommendation**: Use Vercel or Netlify for the best developer experience at zero cost.
