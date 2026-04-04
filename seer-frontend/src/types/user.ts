/**
 * User settings and preferences types
 */

export interface OnboardingData {
  completed: boolean;
  payment_method_added?: boolean;
  discoveryChannel: 'Reddit' | 'YouTube' | 'Twitter' | 'Product Hunt' | 'Google Search' | 'Friend or Colleague' | 'Other';
  experienceLevel: 'Well-versed' | 'Been using for a bit' | 'Just getting started' | 'Completely new';
  integrations?: string[];
  completedAt?: string;
}

export interface UserSettings {
  timezone?: string | null;
  preferences?: {
    onboarding?: OnboardingData;
  };
}

export interface UserSettingsResponse {
  timezone?: string | null;
  preferences?: {
    onboarding?: OnboardingData;
  };
}
