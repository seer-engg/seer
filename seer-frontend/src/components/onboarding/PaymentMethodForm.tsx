import { useMemo, useState } from 'react';
import { CardElement, useStripe, useElements } from '@stripe/react-stripe-js';
import { Button } from '@/components/ui/button';
import { Loader2, CreditCard, Lock } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useTheme } from 'next-themes';

interface PaymentMethodFormProps {
  clientSecret: string;
  trialDays?: number;
  onSuccess: () => void;
  onError: (error: string) => void;
}

const SecurityNotice = () => (
  <div className="flex items-center gap-2 text-sm text-muted-foreground bg-muted/50 p-3 rounded-lg">
    <Lock className="w-4 h-4" />
    <span>Your payment information is encrypted and secure</span>
  </div>
);

const TrialNotice = ({ trialDays }: { trialDays?: number }) => (
  <div className="text-sm text-muted-foreground bg-emerald-500/10 border border-emerald-500/20 p-3 rounded-lg">
    <p className="font-medium text-emerald-600 dark:text-emerald-400">
      {trialDays ? `${trialDays}-Day Free Trial` : 'Free Trial'}
    </p>
    <p className="mt-1">Your card won't be charged until your trial ends. Cancel anytime.</p>
  </div>
);

interface CardInputSectionProps {
  cardComplete: boolean;
  cardElementStyles: object;
  onCardChange: (complete: boolean) => void;
}

const CardInputSection = ({ cardComplete, cardElementStyles, onCardChange }: CardInputSectionProps) => (
  <div className="space-y-2">
    <label className="text-sm font-medium">Card Information</label>
    <div className={cn(
      "border rounded-lg p-3 transition-all",
      cardComplete ? "border-success" : "border-input",
      "focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2"
    )}>
      <CardElement
        options={{
          style: cardElementStyles,
          hidePostalCode: false,
        }}
        onChange={(e) => onCardChange(e.complete)}
      />
    </div>
  </div>
);

export const PaymentMethodForm = ({ clientSecret, trialDays, onSuccess, onError }: PaymentMethodFormProps) => {
  const stripe = useStripe();
  const elements = useElements();
  const { resolvedTheme } = useTheme();
  const [isProcessing, setIsProcessing] = useState(false);
  const [cardComplete, setCardComplete] = useState(false);

  const cardElementStyles = useMemo(() => {
    const isDark = resolvedTheme === 'dark';

    // Stripe elements don't support CSS variables or HSL, so pass explicit hex colors per theme
    const palette = isDark
      ? {
          text: '#e5e7eb', // gray-200
          placeholder: '#9ca3af', // gray-400
          icon: '#d1d5db', // gray-300
        }
      : {
          text: '#111827', // gray-900
          placeholder: '#6b7280', // gray-500
          icon: '#111827', // gray-900
        };

    return {
      base: {
        fontSize: '16px',
        color: palette.text,
        fontFamily: 'inherit',
        iconColor: palette.icon,
        // Stripe elements live in an iframe, so pass explicit colors for dark mode readability
        '::placeholder': { color: palette.placeholder },
      },
      invalid: { color: '#ef4444' }, // red-500
    };
  }, [resolvedTheme]);

  const handleSubmit = async () => {
    console.log('[PaymentMethodForm] handleSubmit called', { clientSecret, hasStripe: !!stripe, hasElements: !!elements, cardComplete });

    if (!stripe || !elements) {
      onError('Stripe is not loaded');
      return;
    }

    const cardElement = elements.getElement(CardElement);
    if (!cardElement) {
      onError('Card element not found');
      return;
    }

    setIsProcessing(true);
    console.log('[PaymentMethodForm] Starting Stripe confirmCardSetup...');

    try {
      // Confirm the Setup Intent with the card element
      const { error, setupIntent } = await stripe.confirmCardSetup(clientSecret, {
        payment_method: {
          card: cardElement,
        },
      });

      console.log('[PaymentMethodForm] confirmCardSetup response:', { error, setupIntent });

      if (error) {
        console.error('Stripe confirmation error:', error);
        onError(error.message || 'Failed to add payment method');
        setIsProcessing(false);
      } else if (setupIntent?.status === 'succeeded') {
        console.log('Setup Intent succeeded:', setupIntent.id);
        onSuccess();
      } else {
        console.error('Unexpected Setup Intent status:', setupIntent?.status);
        onError(`Payment setup incomplete. Status: ${setupIntent?.status || 'unknown'}`);
        setIsProcessing(false);
      }
    } catch (err) {
      console.error('Unexpected error during card confirmation:', err);
      onError('An unexpected error occurred. Please try again.');
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-6">
      <SecurityNotice />
      <CardInputSection
        cardComplete={cardComplete}
        cardElementStyles={cardElementStyles}
        onCardChange={setCardComplete}
      />
      <TrialNotice trialDays={trialDays} />
      <Button
        type="button"
        variant="brand"
        onClick={handleSubmit}
        disabled={!stripe || !cardComplete || isProcessing}
        className="w-full gap-2"
      >
        {isProcessing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Processing...
          </>
        ) : (
          <>
            <CreditCard className="w-4 h-4" />
            Add Payment Method
          </>
        )}
      </Button>
    </div>
  );
};
