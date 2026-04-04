import './wdyr';
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ClerkProvider } from '@clerk/clerk-react'
import { PostHogProvider } from './components/general/PostHogProvider.tsx'
import { GoogleAnalyticsProvider } from './components/general/GoogleAnalyticsProvider.tsx'
import { SentryProvider } from './components/general/SentryProvider.tsx'
import { captureUTMParams } from './utils/utm-tracker'
import { initSentry } from './lib/sentry'
import { initAuthConfig, isLocalAuth } from './hooks/useAuthProvider'

// Initialize Sentry BEFORE React renders to catch early errors
initSentry()

// Capture UTM parameters for signup attribution
captureUTMParams()

const AppTree = () => (
  <SentryProvider>
    <PostHogProvider>
      <GoogleAnalyticsProvider>
        <App />
      </GoogleAnalyticsProvider>
    </PostHogProvider>
  </SentryProvider>
);

// Fetch auth config from backend, then render the app
initAuthConfig().then(() => {
  if (isLocalAuth()) {
    // Local auth mode: no Clerk, no login required
    createRoot(document.getElementById('root')!).render(
      <StrictMode>
        <AppTree />
      </StrictMode>,
    )
  } else {
    // Clerk auth mode: require publishable key
    const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY
    if (!PUBLISHABLE_KEY) {
      throw new Error('AUTH_PROVIDER is set to "clerk" but VITE_CLERK_PUBLISHABLE_KEY is missing from the frontend .env')
    }

    createRoot(document.getElementById('root')!).render(
      <StrictMode>
        <ClerkProvider publishableKey={PUBLISHABLE_KEY}>
          <AppTree />
        </ClerkProvider>
      </StrictMode>,
    )
  }
});
