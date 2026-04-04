/* eslint-disable max-lines, max-lines-per-function */
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient, QueryClient } from "@tanstack/react-query";
import { useForm, Control } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Loader2, Check, ArrowLeft } from "lucide-react";
import { loadStripe, Stripe } from "@stripe/stripe-js";
import { Elements } from "@stripe/react-stripe-js";
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from "@/components/ui/form";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { toast } from "@/components/ui/sonner";
import { backendApiClient } from "@/lib/api-client";
import { userKeys } from "@/lib/query-keys";
import { subscriptionApi } from "@/lib/subscription-api";
import type { OnboardingData } from "@/types/user";
import { cn } from "@/lib/utils";
import { PaymentMethodForm } from "@/components/onboarding/PaymentMethodForm";

const onboardingSchema = z.object({
  discoveryChannel: z.enum([
    'Reddit',
    'YouTube',
    'Twitter',
    'Product Hunt',
    'Google Search',
    'Friend or Colleague',
    'Other'
  ], {
    required_error: "Please select how you discovered Seer",
  }),
  experienceLevel: z.enum([
    'Well-versed',
    'Been using for a bit',
    'Just getting started',
    'Completely new'
  ], {
    required_error: "Please select your experience level",
  }),
  integrations: z.array(z.string()).optional(),
  selectedTier: z.enum(['pro', 'pro_plus']).optional(),
  selectedInterval: z.enum(['month', 'year']).optional(),
});

type OnboardingFormData = z.infer<typeof onboardingSchema>;

const discoveryOptions = [
  { value: 'Reddit', label: 'Reddit' },
  { value: 'YouTube', label: 'YouTube' },
  { value: 'Twitter', label: 'Twitter' },
  { value: 'Product Hunt', label: 'Product Hunt' },
  { value: 'Google Search', label: 'Google Search' },
  { value: 'Friend or Colleague', label: 'Friend or Colleague' },
  { value: 'Other', label: 'Other' },
] as const;

const experienceOptions = [
  {
    value: 'Well-versed',
    label: 'Well-versed',
    description: "I've built many workflows before"
  },
  {
    value: 'Been using for a bit',
    label: 'Been using for a bit',
    description: "I have some experience"
  },
  {
    value: 'Just getting started',
    label: 'Just getting started',
    description: "I'm relatively new to this"
  },
  {
    value: 'Completely new',
    label: 'Completely new',
    description: "This is my first time"
  },
] as const;

const integrationOptions = [
  'Google Workspace (Docs, Sheets, Drive)',
  'Slack',
  'Microsoft Teams',
  'Salesforce',
  'ChatGPT / LLMs',
  'Asana',
  'Linear',
  'GitHub',
  'Jira',
  'Notion',
];

const TOTAL_STEPS = 4;

// Helper Functions
const confirmSetupIntent = async (setupIntentId: string) => {
  return backendApiClient.request('/api/subscriptions/setup-intent/confirm', {
    method: 'POST',
    body: { setup_intent_id: setupIntentId }
  });
};

const saveOnboardingData = async (data: OnboardingFormData, queryClient: QueryClient) => {
  const onboardingData: OnboardingData = {
    completed: true,
    payment_method_added: true,
    discoveryChannel: data.discoveryChannel,
    experienceLevel: data.experienceLevel,
    integrations: data.integrations,
    completedAt: new Date().toISOString(),
  };

  await backendApiClient.request('/api/users/me/settings', {
    method: 'PATCH',
    body: {
      preferences: {
        onboarding: onboardingData,
      },
    },
  });

  await queryClient.invalidateQueries({ queryKey: userKeys.settings() });
};

// Custom Hooks
const useStripeSetup = () => {
  const [stripePromise, setStripePromise] = useState<Promise<Stripe | null> | null>(null);

  useEffect(() => {
    backendApiClient.request<{ publishable_key: string }>('/api/subscriptions/config')
      .then(data => {
        if (data.publishable_key) {
          setStripePromise(loadStripe(data.publishable_key));
        }
      })
      .catch(error => {
        console.error('Failed to load Stripe config:', error);
      });
  }, []);

  return stripePromise;
};

const useSetupIntent = (currentStep: number) => {
  const [clientSecret, setClientSecret] = useState<string | null>(null);
  const [setupIntentId, setSetupIntentId] = useState<string | null>(null);
  const [paymentError, setPaymentError] = useState<string | null>(null);

  useEffect(() => {
    if (currentStep === 4 && !clientSecret) {
      backendApiClient.request<{ client_secret: string }>('/api/subscriptions/setup-intent', {
        method: 'POST'
      })
        .then(data => {
          setClientSecret(data.client_secret);
          const intentId = data.client_secret.split('_secret_')[0];
          setSetupIntentId(intentId);
        })
        .catch(error => {
          console.error('Failed to create setup intent:', error);
          setPaymentError('Failed to initialize payment form. Please try again.');
        });
    }
  }, [currentStep, clientSecret]);

  return { clientSecret, setupIntentId, paymentError, setPaymentError };
};

// Helper Components
interface ProgressStepProps {
  step: number;
  currentStep: number;
}

const ProgressStep = ({ step, currentStep }: ProgressStepProps) => (
  <div
    className={cn(
      "flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all duration-300",
      step < currentStep && "bg-success border-success text-white",
      step === currentStep && "bg-seer border-seer text-white glow-seer",
      step > currentStep && "bg-background border-border text-muted-foreground"
    )}
  >
    {step < currentStep ? (
      <Check className="w-5 h-5" />
    ) : (
      <span className="text-sm font-semibold">{step}</span>
    )}
  </div>
);

interface DiscoveryStepProps {
  control: Control<OnboardingFormData>;
  onSelect: (value: string) => void;
}

const DiscoveryStep = ({ control, onSelect }: DiscoveryStepProps) => (
  <div className="space-y-6 animate-in fade-in duration-300">
    <FormField
      control={control}
      name="discoveryChannel"
      render={({ field }) => (
        <FormItem className="space-y-4">
          <FormLabel className="text-xl font-semibold">
            How did you learn about Seer?
          </FormLabel>
          <FormControl>
            <RadioGroup
              onValueChange={onSelect}
              value={field.value}
              className="flex flex-col space-y-3"
            >
              {discoveryOptions.map((option) => (
                <label
                  key={option.value}
                  htmlFor={`discovery-${option.value}`}
                  className={cn(
                    "flex items-center space-x-3 p-4 rounded-lg border-2 cursor-pointer transition-all hover:border-seer/50 hover:bg-seer/5",
                    field.value === option.value && "border-seer bg-seer/10"
                  )}
                >
                  <RadioGroupItem value={option.value} id={`discovery-${option.value}`} />
                  <span className="text-base font-medium flex-1">
                    {option.label}
                  </span>
                </label>
              ))}
            </RadioGroup>
          </FormControl>
          <FormMessage />
        </FormItem>
      )}
    />
  </div>
);

interface ExperienceStepProps {
  control: Control<OnboardingFormData>;
  onSelect: (value: string) => void;
}

const ExperienceStep = ({ control, onSelect }: ExperienceStepProps) => (
  <div className="space-y-6 animate-in fade-in duration-300">
    <FormField
      control={control}
      name="experienceLevel"
      render={({ field }) => (
        <FormItem className="space-y-4">
          <FormLabel className="text-xl font-semibold">
            How much experience do you have with building workflows?
          </FormLabel>
          <FormControl>
            <RadioGroup
              onValueChange={onSelect}
              value={field.value}
              className="flex flex-col space-y-3"
            >
              {experienceOptions.map((option) => (
                <label
                  key={option.value}
                  htmlFor={`experience-${option.value}`}
                  className={cn(
                    "flex items-start space-x-3 p-4 rounded-lg border-2 cursor-pointer transition-all hover:border-seer/50 hover:bg-seer/5",
                    field.value === option.value && "border-seer bg-seer/10"
                  )}
                >
                  <RadioGroupItem value={option.value} id={`experience-${option.value}`} className="mt-1" />
                  <div className="flex-1">
                    <span className="text-base font-medium block">
                      {option.label}
                    </span>
                    <p className="text-sm text-muted-foreground mt-1">
                      {option.description}
                    </p>
                  </div>
                </label>
              ))}
            </RadioGroup>
          </FormControl>
          <FormMessage />
        </FormItem>
      )}
    />
  </div>
);

interface IntegrationsStepProps {
  control: Control<OnboardingFormData>;
}

const IntegrationsStep = ({ control }: IntegrationsStepProps) => (
  <div className="space-y-6 animate-in fade-in duration-300">
    <FormField
      control={control}
      name="integrations"
      render={() => (
        <FormItem>
          <FormLabel className="text-xl font-semibold">
            What integrations are you interested in?
          </FormLabel>
          <p className="text-sm text-muted-foreground mb-4">
            Select all that apply (optional)
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {integrationOptions.map((integration) => (
              <FormField
                key={integration}
                control={control}
                name="integrations"
                render={({ field }) => {
                  const isChecked = field.value?.includes(integration);
                  return (
                    <FormItem key={integration}>
                      <label
                        className={cn(
                          "flex items-center space-x-3 p-3 rounded-lg border-2 cursor-pointer transition-all hover:border-seer/50 hover:bg-seer/5",
                          isChecked && "border-seer bg-seer/10"
                        )}
                      >
                        <FormControl>
                          <Checkbox
                            checked={isChecked}
                            onCheckedChange={(checked) => {
                              return checked
                                ? field.onChange([...(field.value || []), integration])
                                : field.onChange(
                                    field.value?.filter((value) => value !== integration)
                                  );
                            }}
                          />
                        </FormControl>
                        <span className="text-sm font-medium flex-1">
                          {integration}
                        </span>
                      </label>
                    </FormItem>
                  );
                }}
              />
            ))}
          </div>
          <FormMessage />
        </FormItem>
      )}
    />
  </div>
);



interface OnboardingHeaderProps {
  currentStep: number;
}

const OnboardingHeader = ({ currentStep }: OnboardingHeaderProps) => (
  <CardHeader className="text-center space-y-4 pb-6">
    <CardTitle className="text-3xl font-bold text-gradient-seer">
      Welcome to Seer
    </CardTitle>
    <CardDescription className="text-base">
      Let's personalize your experience
    </CardDescription>

    <div className="flex items-center justify-center gap-2 pt-4">
      {[1, 2, 3, 4].map((step) => (
        <div key={step} className="flex items-center">
          <ProgressStep step={step} currentStep={currentStep} />
          {step < 4 && (
            <div
              className={cn(
                "w-16 h-0.5 mx-1 transition-all duration-300",
                step < currentStep ? "bg-success" : "bg-border"
              )}
            />
          )}
        </div>
      ))}
    </div>

    <p className="text-sm text-muted-foreground">
      Step {currentStep} of {TOTAL_STEPS}
    </p>
  </CardHeader>
);

interface NavigationButtonsProps {
  currentStep: number;
  isSubmitting: boolean;
  onBack: () => void;
  onContinue?: () => void;
}

const NavigationButtons = ({ currentStep, isSubmitting, onBack, onContinue }: NavigationButtonsProps) => {
  if (currentStep === 1) return null;

  return (
    <div className="flex items-center justify-between pt-6 border-t">
      <Button
        type="button"
        variant="ghost"
        onClick={onBack}
        disabled={isSubmitting}
        className="gap-2"
      >
        <ArrowLeft className="w-4 h-4" />
        Back
      </Button>

      {currentStep === 3 && onContinue && (
        <Button
          type="button"
          variant="brand"
          onClick={onContinue}
          className="gap-2"
        >
          Continue
        </Button>
      )}
    </div>
  );
};

interface PlanAndPaymentStepProps {
  control: Control<OnboardingFormData>;
  selectedTier?: 'pro' | 'pro_plus';
  selectedInterval?: 'month' | 'year';
  onPlanSelect: (tier: 'pro' | 'pro_plus', interval: 'month' | 'year') => void;
  stripePromise: Promise<Stripe | null> | null;
  clientSecret: string | null;
  paymentError: string | null;
  onSuccess: () => void;
  onError: (error: string) => void;
}

const PlanAndPaymentStep = ({
  control,
  selectedTier,
  selectedInterval,
  onPlanSelect,
  stripePromise,
  clientSecret,
  paymentError,
  onSuccess,
  onError
}: PlanAndPaymentStepProps) => {
  const [interval, setInterval] = useState<'month' | 'year'>(selectedInterval || 'month');
  const [pricingData, setPricingData] = useState<{
    prices?: Array<{
      tier: string;
      name: string;
      monthly: { price: number; original_price?: number | null; trial_period_days?: number | null };
      annual: { price: number; original_price?: number | null; trial_period_days?: number | null };
      features?: string[];
      badge?: string | null;
    }>;
  } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    subscriptionApi.getPricing()
      .then((data) => {
        setPricingData(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Failed to fetch pricing:', error);
        setLoading(false);
      });
  }, []);

  const plans = pricingData?.prices ?? [];

  const getPrice = (plan: typeof plans[number] | undefined) => {
    if (!plan) return 0;
    return interval === 'month' ? plan.monthly.price : plan.annual.price;
  };

  const getOriginalPrice = (plan: typeof plans[number] | undefined) => {
    if (!plan) return null;
    return interval === 'month' ? plan.monthly.original_price : plan.annual.original_price;
  };

  const getTrialDays = (plan: typeof plans[number] | undefined) => {
    if (!plan) return null;
    return plan.monthly.trial_period_days ?? plan.annual.trial_period_days ?? null;
  };

  const formatPrice = (price: number) => {
    return (price / 100).toFixed(2);
  };

  const handlePlanClick = (tier: 'pro' | 'pro_plus') => {
    onPlanSelect(tier, interval);
  };

  const handleIntervalChange = (newInterval: 'month' | 'year') => {
    setInterval(newInterval);
    if (selectedTier) {
      onPlanSelect(selectedTier, newInterval);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-seer" />
      </div>
    );
  }

  // Derive trial text from the first plan that has trial days
  const firstTrialDays = plans.reduce<number | null>((acc, p) => acc ?? getTrialDays(p), null);
  const trialText = firstTrialDays ? `${firstTrialDays}-day free trial` : 'free trial';

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Plan Selection Section */}
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-xl font-semibold">Choose Your Plan</h2>
          <p className="text-sm text-muted-foreground">
            Start your {trialText}. No charge until trial ends.
          </p>
        </div>

        <div className="flex items-center justify-center gap-3">
          <button
            type="button"
            onClick={() => handleIntervalChange('month')}
            className={cn(
              "px-4 py-2 rounded-lg font-medium transition-all",
              interval === 'month'
                ? "bg-seer text-white"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            )}
          >
            Monthly
          </button>
          <button
            type="button"
            onClick={() => handleIntervalChange('year')}
            className={cn(
              "px-4 py-2 rounded-lg font-medium transition-all",
              interval === 'year'
                ? "bg-seer text-white"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            )}
          >
            Annual
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {plans.map((plan) => {
            const tierKey = plan.tier as 'pro' | 'pro_plus';
            const price = getPrice(plan);
            const originalPrice = getOriginalPrice(plan);
            const trialDays = getTrialDays(plan);
            const features = plan.features ?? [];

            return (
              <Card
                key={plan.tier}
                className={cn(
                  "border-2 cursor-pointer transition-all hover:border-seer/50 hover:shadow-lg relative",
                  selectedTier === tierKey && "border-seer bg-seer/5",
                  plan.badge && selectedTier !== tierKey && "border-seer"
                )}
                onClick={() => handlePlanClick(tierKey)}
              >
                {plan.badge && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <span className="bg-seer text-white text-xs font-bold px-3 py-1 rounded-full">
                      {plan.badge}
                    </span>
                  </div>
                )}
                <CardHeader>
                  <CardTitle className="text-2xl">{plan.name}</CardTitle>
                  <div className="mt-4">
                    <div className="flex items-baseline gap-2">
                      {originalPrice != null && (
                        <span className="text-2xl font-medium text-muted-foreground line-through">
                          ${formatPrice(originalPrice)}
                        </span>
                      )}
                      <span className="text-4xl font-bold">${formatPrice(price)}</span>
                      <span className="text-muted-foreground">/{interval === 'month' ? 'mo' : 'yr'}</span>
                    </div>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {trialDays != null && (
                      <span className="inline-block bg-success/10 text-success text-xs font-semibold px-3 py-1 rounded-full border border-success/20">
                        {trialDays}-day free trial
                      </span>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  {features.length > 0 && (
                    <ul className="space-y-2 text-sm">
                      {features.map((feature) => (
                        <li key={feature} className="flex items-center gap-2">
                          <Check className="w-4 h-4 text-success" />
                          <span>{feature}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Payment Section - Only show when plan is selected */}
      {selectedTier && (
        <div className="space-y-6 border-t pt-6">
          <div>
            <h3 className="text-xl font-semibold mb-2">Add Payment Method</h3>
            <p className="text-sm text-muted-foreground">
              {firstTrialDays
                ? `Your ${firstTrialDays}-day free trial starts now. You won't be charged until the trial ends.`
                : "You won't be charged until the trial ends."
              }
            </p>
          </div>

          {stripePromise && clientSecret ? (
            <Elements stripe={stripePromise}>
              <PaymentMethodForm
                clientSecret={clientSecret}
                trialDays={firstTrialDays ?? undefined}
                onSuccess={onSuccess}
                onError={onError}
              />
            </Elements>
          ) : (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          )}

          {paymentError && (
            <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 p-3 rounded-lg">
              {paymentError}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

const Onboarding = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const stripePromise = useStripeSetup();
  const { clientSecret, setupIntentId, paymentError, setPaymentError } = useSetupIntent(currentStep);

  const form = useForm<OnboardingFormData>({
    resolver: zodResolver(onboardingSchema),
    defaultValues: {
      integrations: [],
    },
    mode: 'onChange',
  });

  const { watch } = form;

  const handlePaymentMethodAdded = async () => {
    console.log('[Onboarding] handlePaymentMethodAdded called', { setupIntentId });

    if (!setupIntentId) {
      setPaymentError('Payment method not properly initialized');
      return;
    }

    const formData = form.getValues();

    if (!formData.selectedTier || !formData.selectedInterval) {
      setPaymentError('Plan selection is required');
      return;
    }

    setIsSubmitting(true);

    try {
      // Confirm the Setup Intent
      await confirmSetupIntent(setupIntentId);

      // Create subscription with trial
      const subscriptionResult = await subscriptionApi.createSubscriptionWithTrial(
        formData.selectedTier,
        formData.selectedInterval
      );

      console.log('[Onboarding] Trial subscription created:', subscriptionResult);

      // Save onboarding data
      await saveOnboardingData(formData, queryClient);

      // Show success message with trial info
      const trialEndDate = subscriptionResult.trial_end
        ? new Date(subscriptionResult.trial_end).toLocaleDateString()
        : null;

      toast.success("Welcome to Seer!", {
        description: trialEndDate
          ? `Your free trial is now active. Trial ends ${trialEndDate}.`
          : "Your subscription is now active.",
      });

      // Navigate to home
      navigate('/', { replace: true });
    } catch (error) {
      console.error('[Onboarding] Failed to complete setup:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to complete setup. Please try again.';
      setPaymentError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  const onSubmit = async (data: OnboardingFormData) => {
    setIsSubmitting(true);
    try {
      await saveOnboardingData(data, queryClient);
      toast.success("Welcome to Seer!", {
        description: "Your preferences have been saved.",
      });
      navigate('/', { replace: true });
    } catch (error) {
      console.error('Failed to save onboarding data:', error);
      toast.error("Failed to save preferences", {
        description: "Please try again.",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleBack = () => currentStep > 1 && setCurrentStep(currentStep - 1);

  const handleDiscoveryChannelChange = (value: string) => {
    form.setValue('discoveryChannel', value as OnboardingFormData['discoveryChannel']);
    setTimeout(() => setCurrentStep(2), 300);
  };

  const handleExperienceLevelChange = (value: string) => {
    form.setValue('experienceLevel', value as OnboardingFormData['experienceLevel']);
    setTimeout(() => setCurrentStep(3), 300);
  };

  const handlePlanSelection = (tier: 'pro' | 'pro_plus', interval: 'month' | 'year') => {
    form.setValue('selectedTier', tier);
    form.setValue('selectedInterval', interval);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-2xl">
        <OnboardingHeader currentStep={currentStep} />
        <CardContent className="min-h-[400px]">
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
              {currentStep === 1 && (
                <DiscoveryStep control={form.control} onSelect={handleDiscoveryChannelChange} />
              )}
              {currentStep === 2 && (
                <ExperienceStep control={form.control} onSelect={handleExperienceLevelChange} />
              )}
              {currentStep === 3 && (
                <IntegrationsStep control={form.control} />
              )}
              {currentStep === 4 && (
                <PlanAndPaymentStep
                  control={form.control}
                  selectedTier={watch('selectedTier')}
                  selectedInterval={watch('selectedInterval')}
                  onPlanSelect={handlePlanSelection}
                  stripePromise={stripePromise}
                  clientSecret={clientSecret}
                  paymentError={paymentError}
                  onSuccess={handlePaymentMethodAdded}
                  onError={setPaymentError}
                />
              )}
              <NavigationButtons
                currentStep={currentStep}
                isSubmitting={isSubmitting}
                onBack={handleBack}
                onContinue={currentStep === 3 ? () => setCurrentStep(4) : undefined}
              />
            </form>
          </Form>
        </CardContent>
      </Card>
    </div>
  );
};

export default Onboarding;
