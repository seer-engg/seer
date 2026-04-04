import { RedirectToSignIn } from "@clerk/clerk-react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { toast } from "@/components/ui/sonner";
import { clearSignupSource } from "../../utils/utm-tracker";
import { backendApiClient } from "@/lib/api-client";
import { userKeys } from "@/lib/query-keys";
import type { UserSettingsResponse } from "@/types/user";
import { useAuthStatus, isLocalAuth } from "@/hooks/useAuthProvider";

interface ProtectedRouteProps {
  children: React.ReactNode;
}

interface OnboardingRedirect {
  showPaymentMessage?: boolean;
  to: "/" | "/onboarding";
}

function getOnboardingRedirect(
  pathname: string,
  userSettings: UserSettingsResponse
): OnboardingRedirect | null {
  const onboardingCompleted = userSettings.preferences?.onboarding?.completed;
  const paymentMethodAdded = userSettings.preferences?.onboarding?.payment_method_added;
  const isOnboardingPage = pathname === "/onboarding";

  const isLocalDev = window.location.hostname === "localhost";

  if (!onboardingCompleted && !isOnboardingPage && !isLocalDev) {
    return { to: "/onboarding" };
  }

  if (onboardingCompleted && !paymentMethodAdded && !isOnboardingPage && !isLocalDev) {
    return { to: "/onboarding", showPaymentMessage: true };
  }

  if (onboardingCompleted && paymentMethodAdded && isOnboardingPage) {
    return { to: "/" };
  }

  return null;
}

const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
  const { isLoaded, isSignedIn } = useAuthStatus();
  const location = useLocation();
  const navigate = useNavigate();

  // Fetch user settings to check onboarding status
  const { data: userSettings, isLoading: isLoadingSettings } = useQuery({
    queryKey: userKeys.settings(),
    queryFn: () => backendApiClient.request<UserSettingsResponse>('/api/users/me/settings'),
    enabled: isLoaded && isSignedIn,
    retry: 1,
  });

  useEffect(() => {
    // If user just signed up (has signup_source in URL), clear it from localStorage
    // and remove from URL to clean up
    const searchParams = new URLSearchParams(location.search);
    if (searchParams.has('signup_source')) {
      clearSignupSource();

      // Remove signup_source from URL
      searchParams.delete('signup_source');
      const newSearch = searchParams.toString();
      const newUrl = location.pathname + (newSearch ? `?${newSearch}` : '');

      // Replace the URL without reloading
      navigate(newUrl, { replace: true });

      console.log('[ProtectedRoute] Signup source captured by backend, cleared from localStorage');
    }
  }, [location, navigate]);

  useEffect(() => {
    if (!isSignedIn || isLoadingSettings || !userSettings) {
      return;
    }

    const redirect = getOnboardingRedirect(location.pathname, userSettings);
    if (!redirect) {
      return;
    }

    navigate(redirect.to, { replace: true });

    if (redirect.showPaymentMessage) {
      toast.error('Please add a payment method to continue');
    }
  }, [isSignedIn, isLoadingSettings, userSettings, location.pathname, navigate]);

  if (!isLoaded || (isSignedIn && isLoadingSettings)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-foreground" />
      </div>
    );
  }

  if (!isSignedIn) {
    return <RedirectToSignIn />;
  }

  return <>{children}</>;
};

export default ProtectedRoute;
