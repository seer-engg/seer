/**
 * InvitationAccept Page
 *
 * Public page for accepting team invitations.
 * Handles both authenticated and unauthenticated states.
 */

import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useUser, SignIn } from '@clerk/clerk-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { Building2, Loader2, AlertCircle, CheckCircle2, XCircle, Users } from 'lucide-react';
import { SeerLogo } from '@/components/icons/seer-logo';
import { organizationApi } from '@/lib/organization-api';
import { useOrganizationStore } from '@/stores/organizationStore';
import type { InvitationDetailsResponse } from '@/types/organization';
import { getRoleDisplayName } from '@/types/organization';

type PageState = 'loading' | 'show_invitation' | 'need_auth' | 'accepting' | 'accepted' | 'declined' | 'error';

/* eslint-disable max-lines-per-function, complexity */
export default function InvitationAccept() {
  const { token } = useParams<{ token: string }>();
  const navigate = useNavigate();
  const { isSignedIn, isLoaded: isUserLoaded } = useUser();

  const [pageState, setPageState] = useState<PageState>('loading');
  const [invitation, setInvitation] = useState<InvitationDetailsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchOrganizations = useOrganizationStore((s) => s.fetchOrganizations);
  const switchOrganization = useOrganizationStore((s) => s.switchOrganization);

  // Fetch invitation details on mount
  useEffect(() => {
    if (!token) {
      setError('Invalid invitation link');
      setPageState('error');
      return;
    }

    const loadInvitation = async () => {
      try {
        const details = await organizationApi.getInvitationByToken(token);
        setInvitation(details);

        if (!isUserLoaded) {
          setPageState('loading');
        } else if (isSignedIn) {
          setPageState('show_invitation');
        } else {
          setPageState('need_auth');
        }
      } catch (err) {
        console.error('Failed to load invitation:', err);
        setError('This invitation is invalid, expired, or has already been used.');
        setPageState('error');
      }
    };

    loadInvitation();
  }, [token, isSignedIn, isUserLoaded]);

  // Update state when auth status changes
  useEffect(() => {
    if (isUserLoaded && invitation) {
      if (isSignedIn) {
        setPageState('show_invitation');
      } else {
        setPageState('need_auth');
      }
    }
  }, [isSignedIn, isUserLoaded, invitation]);

  const handleAccept = async () => {
    if (!token) return;

    setPageState('accepting');
    try {
      const result = await organizationApi.acceptInvitation(token);

      // Refresh organizations and switch to the new one
      await fetchOrganizations();
      await switchOrganization(result.organization.id);

      setPageState('accepted');

      // Redirect after a short delay
      setTimeout(() => {
        navigate('/');
      }, 2000);
    } catch (err) {
      console.error('Failed to accept invitation:', err);
      setError('Failed to accept invitation. Please try again.');
      setPageState('error');
    }
  };

  const handleDecline = async () => {
    if (!token) return;

    try {
      await organizationApi.declineInvitation(token);
      setPageState('declined');

      // Redirect after a short delay
      setTimeout(() => {
        navigate('/');
      }, 2000);
    } catch (err) {
      console.error('Failed to decline invitation:', err);
      setError('Failed to decline invitation. Please try again.');
      setPageState('error');
    }
  };

  // Render helper for different states
  const renderContent = () => {
    switch (pageState) {
      case 'loading':
        return (
          <Card className="w-full max-w-md">
            <CardHeader className="text-center">
              <Skeleton className="h-12 w-12 rounded-full mx-auto mb-2" />
              <Skeleton className="h-6 w-48 mx-auto" />
              <Skeleton className="h-4 w-64 mx-auto mt-2" />
            </CardHeader>
            <CardContent className="space-y-4">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </CardContent>
          </Card>
        );

      case 'need_auth':
        return (
          <div className="w-full max-w-md space-y-6">
            <Card>
              <CardHeader className="text-center">
                <div className="mx-auto mb-2 h-12 w-12 rounded-full bg-seer-500/10 flex items-center justify-center">
                  <Building2 className="h-6 w-6 text-seer-500" />
                </div>
                <CardTitle>Join {invitation?.organizationName}</CardTitle>
                <CardDescription>
                  {invitation?.inviterName} has invited you to join their team
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Alert>
                  <Users className="h-4 w-4" />
                  <AlertDescription className="ml-2">
                    You'll be joining as: <strong>{getRoleDisplayName(invitation?.invitation.role || 'user')}</strong>
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="text-center pb-2">
                <CardDescription>
                  Sign in or create an account to accept this invitation
                </CardDescription>
              </CardHeader>
              <CardContent className="flex justify-center">
                <SignIn
                  routing="hash"
                  afterSignInUrl={`/invitations/${token}`}
                  afterSignUpUrl={`/invitations/${token}`}
                />
              </CardContent>
            </Card>
          </div>
        );

      case 'show_invitation':
        return (
          <Card className="w-full max-w-md">
            <CardHeader className="text-center">
              <div className="mx-auto mb-2 h-12 w-12 rounded-full bg-seer-500/10 flex items-center justify-center">
                <Building2 className="h-6 w-6 text-seer-500" />
              </div>
              <CardTitle>Join {invitation?.organizationName}</CardTitle>
              <CardDescription>
                {invitation?.inviterName} has invited you to join their team
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Alert>
                <Users className="h-4 w-4" />
                <AlertDescription className="ml-2">
                  You'll be joining as: <strong>{getRoleDisplayName(invitation?.invitation.role || 'user')}</strong>
                </AlertDescription>
              </Alert>

              <div className="flex flex-col gap-2">
                <Button onClick={handleAccept} className="w-full">
                  Accept Invitation
                </Button>
                <Button variant="outline" onClick={handleDecline} className="w-full">
                  Decline
                </Button>
              </div>
            </CardContent>
          </Card>
        );

      case 'accepting':
        return (
          <Card className="w-full max-w-md">
            <CardContent className="py-12 text-center">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-seer-500" />
              <p className="text-muted-foreground">Joining team...</p>
            </CardContent>
          </Card>
        );

      case 'accepted':
        return (
          <Card className="w-full max-w-md">
            <CardContent className="py-12 text-center">
              <div className="mx-auto mb-4 h-12 w-12 rounded-full bg-emerald-500/10 flex items-center justify-center">
                <CheckCircle2 className="h-6 w-6 text-emerald-500" />
              </div>
              <h3 className="text-lg font-semibold mb-1">Welcome to the team!</h3>
              <p className="text-muted-foreground">
                You've successfully joined {invitation?.organizationName}
              </p>
              <p className="text-sm text-muted-foreground mt-2">
                Redirecting to dashboard...
              </p>
            </CardContent>
          </Card>
        );

      case 'declined':
        return (
          <Card className="w-full max-w-md">
            <CardContent className="py-12 text-center">
              <div className="mx-auto mb-4 h-12 w-12 rounded-full bg-muted flex items-center justify-center">
                <XCircle className="h-6 w-6 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold mb-1">Invitation Declined</h3>
              <p className="text-muted-foreground">
                You've declined the invitation to join {invitation?.organizationName}
              </p>
              <p className="text-sm text-muted-foreground mt-2">
                Redirecting...
              </p>
            </CardContent>
          </Card>
        );

      case 'error':
        return (
          <Card className="w-full max-w-md">
            <CardContent className="py-12 text-center">
              <div className="mx-auto mb-4 h-12 w-12 rounded-full bg-destructive/10 flex items-center justify-center">
                <AlertCircle className="h-6 w-6 text-destructive" />
              </div>
              <h3 className="text-lg font-semibold mb-1">Invalid Invitation</h3>
              <p className="text-muted-foreground">{error}</p>
              <Button asChild variant="outline" className="mt-4">
                <Link to="/">Go to Dashboard</Link>
              </Button>
            </CardContent>
          </Card>
        );
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
      <div className="mb-8 flex items-center gap-2">
        <SeerLogo className="h-8 w-8 text-primary" />
        <span className="text-xl font-semibold">Seer</span>
      </div>

      {renderContent()}
    </div>
  );
}
