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

// Initialize Sentry BEFORE React renders to catch early errors
initSentry()

// Import your Publishable Key
const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

if (!PUBLISHABLE_KEY) {
throw new Error('Add your Clerk Publishable Key to the .env file')
}

// Capture UTM parameters for signup attribution
captureUTMParams()

createRoot(document.getElementById('root')!).render(
<StrictMode>
    <ClerkProvider publishableKey={PUBLISHABLE_KEY}>
    <SentryProvider>
        <PostHogProvider>
        <GoogleAnalyticsProvider>
            <App />
        </GoogleAnalyticsProvider>
        </PostHogProvider>
    </SentryProvider>
    </ClerkProvider>
</StrictMode>,
)