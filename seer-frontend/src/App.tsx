import { useEffect } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import { AnimatePresence } from "framer-motion";
import Settings from "./pages/Settings";
import NotFound from "./pages/NotFound";
import ProtectedRoute from "./components/general/ProtectedRoute";
import { SeerLayout } from "./components/seer/SeerLayout";
import Workflows from "./pages/Workflows";
import Dashboard from "./pages/Dashboard";
import Onboarding from "./pages/Onboarding";
import { ExecutionTrace } from "./pages/ExecutionTrace";
import { InterruptResume } from "./pages/InterruptResume";
import Templates from "./pages/Templates";
import Triggers from "./pages/Triggers";
import Meetings from "./pages/Meetings";
import { SHOW_MEETINGS } from "./lib/feature-flags";
import DataStorage from "./pages/DataStorage";
import Chat from "./pages/Chat";
import Browser from "./pages/Browser";
import PublicForm from "./pages/PublicForm";
import PublicReplay from "./pages/PublicReplay";
import EmbedWorkflow from "./pages/EmbedWorkflow";
import InvitationAccept from "./pages/InvitationAccept";
import { BillingSettings } from "./pages/settings/BillingSettings";
import './App.css'
import { SignIn, SignUp, useClerk } from "@clerk/clerk-react";
import { getStoredSignupSource } from "./utils/utm-tracker";
import { KeyboardShortcutProvider } from "@/hooks/utility/useKeyboardShortcuts";
import { StoreInitializer } from "@/components/general/StoreInitializer";
import { PaymentRequiredModal } from "@/components/error/PaymentRequiredModal";
import { SentryErrorBoundary } from "@/components/error/SentryErrorBoundary";
import { SentryRouteTracker } from "@/components/general/SentryRouteTracker";
import { AppUpdateBanner } from "@/components/general/AppUpdateBanner";
import { UserEmulationPanel } from "@/components/dev/UserEmulationPanel";
import { OrgCollaborationProvider } from "@/components/general/OrgCollaborationProvider";
import { queryClient } from "@/lib/query-client";

function SignOutPage() {
  const { signOut } = useClerk();
  useEffect(() => { signOut({ redirectUrl: "/sign-in" }); }, [signOut]);
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <p className="text-muted-foreground">Signing out...</p>
    </div>
  );
}

const PUBLIC_PREFIXES = ['/embed/', '/forms/', '/replay/', '/invitations/', '/sign-in', '/sign-up', '/sign-out'];

function AppProviders() {
  const { pathname } = useLocation();
  const isPublicRoute = PUBLIC_PREFIXES.some(p => pathname.startsWith(p));
  if (isPublicRoute) return null;
  return (
    <>
      <StoreInitializer />
      <OrgCollaborationProvider />
      <PaymentRequiredModal />
    </>
  );
}

/* eslint-disable max-lines-per-function */
function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <Dashboard />
              </SeerLayout>
            </ProtectedRoute>
          }
        />

        <Route
          path="/templates"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <Templates />
              </SeerLayout>
            </ProtectedRoute>
          }
        />

        <Route
          path="/templates/:slug"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <Templates />
              </SeerLayout>
            </ProtectedRoute>
          }
        />

        <Route
          path="/triggers"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <Triggers />
              </SeerLayout>
            </ProtectedRoute>
          }
        />

        {SHOW_MEETINGS && (
          <Route
            path="/meetings"
            element={
              <ProtectedRoute>
                <SeerLayout>
                  <Meetings />
                </SeerLayout>
              </ProtectedRoute>
            }
          />
        )}

        <Route
          path="/data"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <DataStorage />
              </SeerLayout>
            </ProtectedRoute>
          }
        />

        <Route
          path="/browser"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <Browser />
              </SeerLayout>
            </ProtectedRoute>
          }
        />

        <Route
          path="/chat"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <Chat />
              </SeerLayout>
            </ProtectedRoute>
          }
        />

        <Route
          path="/chat/:sessionId"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <Chat />
              </SeerLayout>
            </ProtectedRoute>
          }
        />

        <Route
          path="/sign-in/*"
          element={
            <div className="min-h-screen flex items-center justify-center bg-background p-4">
              <SignIn routing="path" path="/sign-in" />
            </div>
          }
        />
        <Route path="/sign-out" element={<SignOutPage />} />
        <Route
          path="/sign-up/*"
          element={
            <div className="min-h-screen flex items-center justify-center bg-background p-4">
              <SignUp
                routing="path"
                path="/sign-up"
                afterSignUpUrl={`/onboarding?signup_source=${getStoredSignupSource() || 'Direct'}`}
              />
            </div>
          }
        />
        <Route
          path="/onboarding"
          element={
            <ProtectedRoute>
              <Onboarding />
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <Settings />
              </SeerLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings/billing"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <BillingSettings />
              </SeerLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/workflows/:workflowId"
          element={
            <ProtectedRoute>
              <SeerLayout defaultOpen={false}>
                <Workflows />
              </SeerLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/workflows"
          element={
            <ProtectedRoute>
              <SeerLayout defaultOpen={false}>
                <Workflows />
              </SeerLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/executions/:runId"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <ExecutionTrace />
              </SeerLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/interrupts/:runId"
          element={
            <ProtectedRoute>
              <SeerLayout>
                <InterruptResume />
              </SeerLayout>
            </ProtectedRoute>
          }
        />

        {/* Public routes */}
        <Route path="/forms/:suffix" element={<PublicForm />} />
        <Route path="/replay/:recordingId" element={<PublicReplay />} />
        <Route path="/embed/workflow/:slug" element={<EmbedWorkflow />} />
        <Route path="/invitations/:token" element={<InvitationAccept />} />

        <Route path="*" element={<NotFound />} />
      </Routes>
    </AnimatePresence>
  );
}

const App = () => (
  <SentryErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
        <TooltipProvider>
          <KeyboardShortcutProvider>
            <Toaster />
            <Sonner />
            <BrowserRouter>
              <SentryRouteTracker />
              <AppProviders />
              <AppUpdateBanner />
              {import.meta.env.VITE_ENABLE_USER_EMULATION === "true" && <UserEmulationPanel />}
              <AnimatedRoutes />
            </BrowserRouter>
          </KeyboardShortcutProvider>
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </SentryErrorBoundary>
);

export default App;
